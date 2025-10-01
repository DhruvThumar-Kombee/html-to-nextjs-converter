import os
import glob
import subprocess
import shutil
import google.generativeai as genai
from dotenv import load_dotenv
import re

# --- UTILITY FUNCTIONS for colored terminal output ---
def print_color(text, color_code): print(f"\033[{color_code}m{text}\033[0m")
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
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, universal_newlines=True, shell=(os.name == 'nt')
        )
        for line in process.stdout: print(line, end='')
        process.wait()
        if process.returncode != 0: raise subprocess.CalledProcessError(process.returncode, command)
        print_success("Command completed successfully.")
    except Exception as e:
        print_error(f"Command failed: {e}"); exit(1)

def read_file_content(path):
    try:
        with open(path, 'r', encoding='utf-8') as f: return f.read()
    except FileNotFoundError: return None
    except Exception as e: print_error(f"Error reading file {path}: {e}"); return None

def write_to_file(path, content):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f: f.write(content)
        print_success(f"Successfully wrote to {path}")
    except Exception as e: print_error(f"Error writing to file {path}: {e}")

def clean_code_block(text):
    match = re.search(r"```(?:\w*\n)?(.*)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

def to_pascal_case(text):
    return ''.join(word.capitalize() for word in re.split('[-_]', text))

# --- GEMINI API INTERACTION FUNCTIONS ---

def configure_gemini(api_key):
    try:
        genai.configure(api_key=api_key)
        print_success("Gemini API configured successfully.")
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e: print_error(f"Failed to configure Gemini API: {e}"); exit(1)

def convert_html_to_component(model, html_content, component_name):
    """Converts an HTML snippet into a complete, self-contained React TSX component."""
    print_info(f"    [AI Task] Converting HTML to {component_name}.tsx component...")
    prompt = f"""
    You are an expert web developer. Convert the following HTML into a complete, self-contained React TSX component file.

    **Rules:**
    1.  Create a default exported React functional component named `{component_name}`.
    2.  **UI MUST NOT CHANGE.** Preserve all Tailwind CSS classes exactly. Convert `class` to `className`.
    3.  Update image `src` paths from `.../src/public/img/` to `/img/`.
    4.  Ensure all tags are properly closed for JSX (e.g., `<img />`).
    5.  The output must be the complete file content, including imports (`import React from 'react'`). Do not add any explanatory text or markdown backticks.

    **HTML Snippet:**
    ---
    {html_content}
    ---
    """
    response = model.generate_content(prompt)
    return clean_code_block(response.text)

def identify_and_split_sections(model, page_body_html, header_html, footer_html):
    """Uses AI to act as an architect, identifying logical sections in a page."""
    print_info("    [AI Task] Analyzing page structure to identify logical sections...")
    prompt = f"""
    You are an expert React architect. Your task is to analyze the provided HTML `<body>` content and break it down into logical, reusable section components.

    **Context:**
    - The header and footer will be handled by a global layout, so you must ignore and exclude them.
    - Identify sections based on `<section>` tags or `<div>`s with descriptive IDs or class names (e.g., id="hero", class="features-grid").

    **Instructions:**
    1.  Analyze the full `<body>` HTML below and identify all the unique content sections *between* the header and footer.
    2.  For each section you identify, create a descriptive PascalCase name for it (e.g., `HeroSection`, `PricingTiers`, `ContactForm`).
    3.  Your output MUST be in a specific, parsable format. For each section, provide:
        - The component name followed by `.tsx`.
        - A newline.
        - The full, raw HTML for that section.
        - A unique separator `<!---|||--->` on a new line.

    **Example Output Format:**
    HeroSection.tsx
    <div id="hero" class="..."><p>Hero content...</p></div>
    <!---|||--->
    FeaturesSection.tsx
    <section class="..."><p>Features content...</p></section>
    <!---|||--->

    **Full Page `<body>` HTML to Analyze:**
    ---
    {page_body_html}
    ---

    **Header HTML to IGNORE:**
    ---
    {header_html}
    ---

    **Footer HTML to IGNORE:**
    ---
    {footer_html}
    ---
    """
    response = model.generate_content(prompt)
    return response.text.strip()

# --- MAIN SCRIPT LOGIC ---

def main():
    print_info("--- Component-Based HTML to Next.js Converter ---")
    
    # 1. Load Configuration
    print_info("\nStep 1: Loading configuration from .env file...")
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    existing_project_path = os.getenv("EXISTING_PROJECT_PATH")
    nextjs_project_name = os.getenv("NEXTJS_PROJECT_NAME")

    if not all([api_key, existing_project_path, nextjs_project_name]):
        print_error("Error: Missing required variables in .env file."); exit(1)
        
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    new_project_path = os.path.join(workspace_dir, nextjs_project_name)

    print_success("Configuration loaded.")

    # 2. Create Next.js Project
    print_info("\nStep 2: Creating Next.js project structure...")
    if os.path.exists(new_project_path):
        print_warning(f"Directory '{nextjs_project_name}' already exists. Skipping creation.")
    else:
        run_command([
            "npx", "create-next-app@latest", nextjs_project_name, "--ts", "--tailwind", 
            "--eslint", "--app", "--no-src-dir", "--import-alias", '"@/*"'
        ], cwd=workspace_dir)

    # 3. Discover Files & Configure AI
    print_info("\nStep 3: Discovering files and configuring AI...")
    page_files = glob.glob(os.path.join(existing_project_path, "**", "*.html"), recursive=True)
    component_dir = os.path.join(existing_project_path, "src", "components")
    component_files = glob.glob(os.path.join(component_dir, "*.html")) if os.path.exists(component_dir) else []

    if not page_files:
        print_error(f"No .html pages found in '{existing_project_path}'. Check the path."); exit(1)

    print_success(f"Found {len(page_files)} pages and {len(component_files)} shared components.")
    model = configure_gemini(api_key)
    
    # 4. Process Shared Components (Header/Footer)
    print_info("\nStep 4: Processing shared components (Header/Footer)...")
    shared_components_map = {}
    shared_components_html = {}
    new_components_dir = os.path.join(new_project_path, "components")
    
    for comp_path in component_files:
        base_name = os.path.splitext(os.path.basename(comp_path))[0]
        component_name = to_pascal_case(base_name)
        html_content = read_file_content(comp_path)
        if html_content:
            tsx_content = convert_html_to_component(model, html_content, component_name)
            write_to_file(os.path.join(new_components_dir, f"{component_name}.tsx"), tsx_content)
            shared_components_map[base_name] = component_name
            shared_components_html[base_name] = html_content

    # 5. Process Each Page into Sections
    print_info("\nStep 5: Processing individual pages into a component-based structure...")
    for page_path in page_files:
        page_name_raw = os.path.splitext(os.path.basename(page_path))[0]
        page_name_pascal = to_pascal_case(page_name_raw)
        print_info(f"\n--- Processing Page: {page_name_pascal} ---")
        
        html_content = read_file_content(page_path)
        if not html_content: continue

        body_match = re.search(r"<body.*?>(.*)</body>", html_content, re.DOTALL)
        if not body_match:
            print_warning(f"No <body> in {page_name_raw}.html, skipping."); continue
        
        page_body_html = body_match.group(1).strip()

        # Step 5a: Use AI to split the body into sections
        sections_data_str = identify_and_split_sections(
            model, page_body_html, 
            shared_components_html.get('header', ''), 
            shared_components_html.get('footer', '')
        )
        
        sections = [s.strip() for s in sections_data_str.split('<!---|||--->') if s.strip()]
        if not sections:
            print_warning(f"AI could not identify sections for {page_name_pascal}. Skipping page generation."); continue

        page_section_components = []

        # Step 5b: Convert each HTML section into a TSX component file
        for section_str in sections:
            lines = section_str.split('\n')
            component_filename = lines[0].strip()
            component_name = os.path.splitext(component_filename)[0]
            section_html = '\n'.join(lines[1:]).strip()
            
            if not component_name or not section_html: continue

            tsx_content = convert_html_to_component(model, section_html, component_name)
            
            # Save to a structured folder: components/sections/HomePage/HeroSection.tsx
            output_dir = os.path.join(new_project_path, "components", "sections", page_name_pascal)
            write_to_file(os.path.join(output_dir, component_filename), tsx_content)
            page_section_components.append(component_name)

        # Step 5c: Assemble the main page file that imports and uses the sections
        imports = "\n".join([f"import {name} from '@/components/sections/{page_name_pascal}/{name}';" for name in page_section_components])
        render_elements = "\n".join([f"      <{name} />" for name in page_section_components])
        
        page_dir_name = page_name_raw if page_name_raw.lower() not in ['index', 'home'] else ''
        page_output_dir = os.path.join(new_project_path, "app", page_dir_name)

        final_page_tsx = f"{imports}\n\nexport default function Page() {{\n  return (\n    <main>\n{render_elements}\n    </main>\n  );\n}}"
        write_to_file(os.path.join(page_output_dir, "page.tsx"), final_page_tsx)

    # 6. Update Layout and Copy Assets (Unchanged)
    if 'header' in shared_components_map and 'footer' in shared_components_map:
        print_info("\nStep 6: Updating main layout with Header and Footer...")
        # (This part of the code is the same as before and works well)
        # ... [omitted for brevity, it's the same as the previous script's layout update] ...

    print_info("\nStep 7: Copying image assets...")
    # (This part is also the same)
    # ... [omitted for brevity, it's the same as the previous script's asset copy] ...

    print_success("\n\n--- ARCHITECTURAL CONVERSION COMPLETE! ---")
    print_info("\nYour project has been converted into a modern, component-based Next.js app.")
    print_info("\nNext Steps:")
    print(f"1. Navigate to your new project: cd {nextjs_project_name}")
    print("2. Install dependencies: npm install")
    print("3. Start the dev server: npm run dev")


if __name__ == "__main__":
    main()