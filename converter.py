import os
import glob
import subprocess
import shutil
import google.generativeai as genai
from dotenv import load_dotenv
import re

# --- UTILITY FUNCTIONS for colored terminal output ---
def print_color(text, color_code):
    print(f"\033[{color_code}m{text}\033[0m")
def print_success(text): print_color(text, "92") # Green
def print_info(text): print_color(text, "94") # Blue
def print_warning(text): print_color(text, "93") # Yellow
def print_error(text): print_color(text, "91") # Red

# --- CORE SCRIPT FUNCTIONS ---

def run_command(command, cwd):
    """Runs a shell command and streams its output."""
    print_info(f"\n> Running command: {' '.join(command)}")
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            shell=True if os.name == 'nt' else False
        )
        for line in process.stdout: print(line, end='')
        process.wait()
        if process.returncode != 0: raise subprocess.CalledProcessError(process.returncode, command)
        print_success("Command completed successfully.")
    except Exception as e:
        print_error(f"Command failed: {e}")
        exit(1)

def read_file_content(path):
    """Reads and returns the content of a file."""
    try:
        with open(path, 'r', encoding='utf-8') as f: return f.read()
    except FileNotFoundError: return None
    except Exception as e:
        print_error(f"Error reading file {path}: {e}")
        return None

def write_to_file(path, content):
    """Writes content to a file, creating directories if needed."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f: f.write(content)
        print_success(f"Successfully wrote to {path}")
    except Exception as e: print_error(f"Error writing to file {path}: {e}")

def clean_code_block(text):
    """Removes markdown code block formatting (```tsx ... ```) from Gemini's response."""
    match = re.search(r"```(?:\w*\n)?(.*)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

def to_pascal_case(text):
    """Converts a string from snake_case or kebab-case to PascalCase."""
    return ''.join(word.capitalize() for word in re.split('[-_]', text))

# --- GEMINI API INTERACTION FUNCTIONS ---

def configure_gemini(api_key):
    """Configures the Gemini API."""
    try:
        genai.configure(api_key=api_key)
        print_success("Gemini API configured successfully.")
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        print_error(f"Failed to configure Gemini API: {e}")
        exit(1)

def convert_shared_component_tsx(model, html_content, component_name):
    """Converts an HTML snippet into a complete, self-contained React TSX component file."""
    print_info(f"\n[AI Task] Converting shared component: {component_name}.tsx...")
    prompt = f"""
    You are an expert web developer. Your task is to convert the following HTML snippet into a complete, reusable React component using TypeScript and Next.js.

    **Rules:**
    1.  **Component Name:** The component must be a default export named `{component_name}`.
    2.  **UI Preservation:** Do NOT change any Tailwind CSS classes. Convert `class` to `className`.
    3.  **Image Paths:** Convert image `src` paths like `src/public/img/logo.png` to `/img/logo.png`.
    4.  **JSX Syntax:** Ensure all tags are properly closed (e.g., `<img />`).
    5.  **Complete File:** Your output must be the complete content for a `.tsx` file, including `import React from 'react';` and the default export. Do not add any explanatory text.

    **HTML Snippet to Convert:**
    ---
    {html_content}
    ---
    """
    response = model.generate_content(prompt)
    return clean_code_block(response.text)

def convert_page_html_to_tsx(model, page_body_html, shared_components_html):
    """Converts the unique content of a page, excluding shared header/footer."""
    print_info(f"\n[AI Task] Converting main page content...")
    header_html = shared_components_html.get('header', '')
    footer_html = shared_components_html.get('footer', '')

    prompt = f"""
    You are an expert web developer. Your task is to convert the main content of an HTML `<body>` into a React JSX fragment for a Next.js page.

    **Context:**
    This page uses a shared layout that will automatically include the header and footer. You MUST NOT include the header or footer in your output.

    **Shared Header HTML (for context, do not include in output):**
    ---
    {header_html}
    ---

    **Shared Footer HTML (for context, do not include in output):**
    ---
    {footer_html}
    ---

    **Full Page `<body>` HTML to Convert:**
    ---
    {page_body_html}
    ---

    **Instructions:**
    1.  **Extract Unique Content:** Analyze the full page `<body>` and identify the content that is *between* the header and the footer.
    2.  **Convert to JSX:** Convert only this unique middle content to JSX.
    3.  **UI Preservation:** Do NOT change any Tailwind CSS classes. Convert `class` to `className`.
    4.  **Image Paths:** Convert `src` paths from `src/public/img/...` to `/img/...`.
    5.  **JSX Syntax:** Ensure all tags are properly closed.
    6.  **Output:** Provide ONLY the raw JSX for the unique page content. Do not include a function definition or imports.
    """
    response = model.generate_content(prompt)
    return clean_code_block(response.text)

# --- MAIN SCRIPT LOGIC ---

def main():
    print_info("--- Advanced HTML to Next.js Converter ---")
    
    # 1. Load Configuration
    print_info("\nStep 1: Loading configuration from .env file...")
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    existing_project_path = os.getenv("EXISTING_PROJECT_PATH")
    nextjs_project_name = os.getenv("NEXTJS_PROJECT_NAME")

    if not all([api_key, existing_project_path, nextjs_project_name]):
        print_error("Error: Missing required variables in .env file.")
        exit(1)
        
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    new_project_path = os.path.join(workspace_dir, nextjs_project_name)

    print_success("Configuration loaded.")

    # 2. Create Next.js Project
    print_info("\nStep 2: Creating Next.js project structure...")
    if os.path.exists(new_project_path):
        print_warning(f"Directory '{nextjs_project_name}' already exists. Skipping creation.")
    else:
        create_command = [
            "npx", "create-next-app@latest", nextjs_project_name,
            "--ts", "--tailwind", "--eslint", "--app",
            "--no-src-dir", "--import-alias", '"@/*"'
        ]
        run_command(create_command, cwd=workspace_dir)

    # 3. Discover HTML Pages and Shared Components
    print_info("\nStep 3: Discovering pages and components...")
    
    print_info(f"Searching for .html files recursively inside: {existing_project_path}")
    page_files = glob.glob(os.path.join(existing_project_path, "**", "*.html"), recursive=True)
    component_dir = os.path.join(existing_project_path, "src", "components")
    component_files = glob.glob(os.path.join(component_dir, "*.html")) if os.path.exists(component_dir) else []

    if not page_files:
        print_error(f"No .html pages found in '{existing_project_path}'. Please check your path.")
        exit(1)

    print_success(f"Found {len(page_files)} pages.")
    print_success(f"Found {len(component_files)} shared components.")

    model = configure_gemini(api_key)
    
    # 4. Convert and Save Shared Components
    print_info("\nStep 4: Processing shared components...")
    shared_components_map = {}
    shared_components_html = {}
    new_components_dir = os.path.join(new_project_path, "components")
    os.makedirs(new_components_dir, exist_ok=True)

    for comp_path in component_files:
        base_name = os.path.splitext(os.path.basename(comp_path))[0]
        component_name = to_pascal_case(base_name)
        html_content = read_file_content(comp_path)
        if html_content:
            tsx_content = convert_shared_component_tsx(model, html_content, component_name)
            write_to_file(os.path.join(new_components_dir, f"{component_name}.tsx"), tsx_content)
            shared_components_map[base_name] = component_name
            shared_components_html[base_name] = html_content

    # 5. Convert and Save Pages
    print_info("\nStep 5: Processing pages...")
    for page_path in page_files:
        page_name = os.path.splitext(os.path.basename(page_path))[0]
        html_content = read_file_content(page_path)
        if not html_content: continue

        body_match = re.search(r"<body.*?>(.*)</body>", html_content, re.DOTALL)
        if not body_match:
            print_warning(f"No <body> tag found in {page_name}.html, skipping.")
            continue
        
        page_body_html = body_match.group(1).strip()
        page_tsx_content = convert_page_html_to_tsx(model, page_body_html, shared_components_html)
        
        # Determine output path (e.g., index.html -> app/page.tsx, about.html -> app/about/page.tsx)
        output_dir = os.path.join(new_project_path, "app")
        if page_name.lower() not in ['index', 'home']:
            output_dir = os.path.join(output_dir, page_name)
        
        final_page_tsx = f"export default function Page() {{\n  return (\n    <>\n      {page_tsx_content}\n    </>\n  );\n}}"
        write_to_file(os.path.join(output_dir, "page.tsx"), final_page_tsx)
    
    # 6. Update Main Layout with Header and Footer
    if 'header' in shared_components_map and 'footer' in shared_components_map:
        print_info("\nStep 6: Updating main layout with Header and Footer...")
        layout_path = os.path.join(new_project_path, "app", "layout.tsx")
        layout_content = read_file_content(layout_path)
        
        header_name = shared_components_map['header']
        footer_name = shared_components_map['footer']

        # Add imports
        imports = f"import {header_name} from '@/components/{header_name}';\nimport {footer_name} from '@/components/{footer_name}';\n"
        layout_content = re.sub(r"(import './globals.css';)", f"\\1\n{imports}", layout_content)
        
        # Add components around children
        body_replacement = f"<body>\n        <{header_name} />\n        {{children}}\n        <{footer_name} />\n      </body>"
        layout_content = re.sub(r"<body>(.*?)</body>", body_replacement, layout_content, flags=re.DOTALL)

        write_to_file(layout_path, layout_content)
    
    # 7. Copy Assets
    print_info("\nStep 7: Copying image assets...")
    source_assets_path = os.path.join(existing_project_path, "src", "public", "img")
    dest_assets_path = os.path.join(new_project_path, "public", "img")
    if os.path.exists(source_assets_path):
        shutil.copytree(source_assets_path, dest_assets_path, dirs_exist_ok=True)
        print_success("Assets copied successfully.")
    else:
        print_warning("Asset directory not found, skipping.")

    print_success("\n\n--- CONVERSION COMPLETE! ---")
    print_info("\nNext Steps:")
    print(f"1. Navigate to your new project: cd {nextjs_project_name}")
    print("2. Install dependencies: npm install")
    print("3. Start the dev server: npm run dev")

if __name__ == "__main__":
    main()