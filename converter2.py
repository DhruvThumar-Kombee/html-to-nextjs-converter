import os
import glob
import subprocess
import shutil
import google.generativeai as genai
from dotenv import load_dotenv
import re
import json

# --- UTILITY FUNCTIONS for colored terminal output ---
def print_color(text, color_code): print(f"\033[{color_code}m{text}\033[0m")
def print_success(text): print_color(text, "92") # Green
def print_info(text): print_color(text, "94") # Blue
def print_warning(text): print_color(text, "93") # Yellow
def print_error(text): print_color(text, "91") # Red

# --- CORE SCRIPT FUNCTIONS ---

def run_command(command, cwd):
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
    except: return None

def write_to_file(path, content):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f: f.write(content)
        print_success(f"Successfully wrote to {path}")
    except Exception as e: print_error(f"Error writing to file {path}: {e}")

def clean_code_block(text, lang=""):
    match = re.search(fr"```{lang}\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

def to_pascal_case(text):
    return ''.join(word.capitalize() for word in re.split('[-_]', text))

# --- GEMINI API INTERACTION FUNCTIONS ---

def configure_gemini(api_key):
    try:
        genai.configure(api_key=api_key)
        print_success("Gemini API configured successfully.")
        return genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e: print_error(f"Failed to configure Gemini API: {e}"); exit(1)

def identify_and_split_sections_ai(model, page_body_html):
    print_info("    [AI Task] Analyzing page structure to identify logical sections...")
    prompt = f"""
    You are an expert HTML analyst. Your task is to analyze the provided HTML body content and split it into a structured list of logical sections.

    **Instructions:**
    1.  Identify major sections. A section is typically a `<section>` tag or a top-level `<div>` with a clear purpose.
    2.  For each section, create a descriptive name in `PascalCase` (e.g., `HeroSection`, `PricingTiers`).
    3.  Your output **MUST** be a valid JSON array of objects. Each object must have two keys: `name` (the PascalCase name) and `html` (the raw HTML content of that section).
    4.  Ensure the HTML for each section is complete and self-contained.
 
    **HTML Body to Analyze:**
    ---
    {page_body_html}
    ---
    """
    response = model.generate_content(prompt)
    try:
        cleaned_response = clean_code_block(response.text, "json")
        return json.loads(cleaned_response)
    except (json.JSONDecodeError, IndexError) as e:
        print_error(f"Failed to parse AI response as JSON for section splitting: {e}")
        return [{"name": "MainContent", "html": page_body_html}]

def convert_html_to_component(model, section_html, component_name, main_js_content):
    print_info(f"    [AI Task] Converting {component_name} into an intelligent React component...")
    
    # Conditionally add instructions about interactivity
    interactivity_prompt = ""
    if main_js_content:
        interactivity_prompt = f"""
    **Client-Side Interactivity:**
    -   Analyze the provided "Full Site JavaScript" below.
    -   If you find any functionality (event listeners, DOM manipulation) that applies specifically to this HTML snippet, you MUST convert that logic into modern React interactivity.
    -   To do this, you MUST add `"use client";` at the very top of the component file.
    -   Use React hooks like `useState`, `useEffect`, and `useRef` to replicate the original functionality.
    -   If no JavaScript logic applies to this specific HTML, generate it as a standard Server Component (NO `"use client";`).
    """
    
    prompt = f"""
    You are an expert Next.js 15 developer building a professional, data-driven application.
    Your task is to convert an HTML snippet into a single, intelligent, and complete React component file.

    **CRITICAL RULES:**
    1.  **UI PRESERVATION:** This is the most important rule. The final rendered output MUST look identical to the original HTML. Do NOT change any Tailwind CSS classes.
    2.  **Next.js 15 Syntax**: All code must use the latest Next.js 15 features and syntax.

    3.  **Data-Driven Props:**
        -   Extract ALL static content (text, titles, button labels, image `src` URLs, etc.) into a TypeScript `interface` for the component's props.
        -   The component must accept these props and use them to render the JSX. This makes the component reusable and separates data from presentation.
    4.  **Convert `<a>` to `<Link>`:** Replace ALL `<a href="...">` tags with `<Link href="...">` from `next/link`. You MUST import `Link` at the top of the file.
    {interactivity_prompt}
    5.  **Generate JSON Data:** Create a complete JSON object containing all the extracted static data that perfectly matches the Props interface you defined.
    6.  **Output Format:** Your response MUST be in two parts, separated by the unique delimiter `<!---|||--->`. First, the complete `.tsx` code. Second, the complete JSON data object.

    **Full Site JavaScript (for context):**
    ---
    {main_js_content or "/* No JavaScript provided. */"}
    ---
    
    **HTML Snippet to Convert:**
    ---
    {section_html}
    ---
    """
    response = model.generate_content(prompt)
    parts = response.text.split('<!---|||--->')
    if len(parts) == 2:
        tsx_content = clean_code_block(parts[0], "tsx")
        json_data_str = clean_code_block(parts[1], "json")
        return tsx_content, json_data_str
    else:
        print_warning(f"AI did not return valid two-part data for {component_name}. Using fallback.")
        return clean_code_block(response.text, "tsx"), "{}"

# --- MAIN SCRIPT LOGIC ---
def main():
    print_info("--- Intelligent Next.js 15 Converter (with In-Component Interactivity) ---")
    
    # 1. Load Config
    load_dotenv()
    api_key, existing_project_path, nextjs_project_name = (
        os.getenv("GOOGLE_API_KEY"), os.getenv("EXISTING_PROJECT_PATH"), os.getenv("NEXTJS_PROJECT_NAME")
    )
    if not all([api_key, existing_project_path, nextjs_project_name]):
        print_error("Error: Missing required variables in .env file."); exit(1)
    
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    new_project_path = os.path.join(workspace_dir, nextjs_project_name)
    print_success("Configuration loaded.")

    # 2. Create Next.js Project with --src-dir
    if not os.path.exists(new_project_path):
        run_command([
            "npx", "create-next-app@latest", nextjs_project_name, "--ts", "--tailwind", 
            "--eslint", "--app", "--src-dir", "--import-alias", '"@/*"'
        ], cwd=workspace_dir)
        # Other installations can go here
    else:
        print_warning(f"Directory '{nextjs_project_name}' already exists. Skipping creation.")

    model = configure_gemini(api_key)
    
    # 3. Read Interactivity File (main.js)
    main_js_path = os.path.join(existing_project_path, "src", "main.js")
    main_js_content = read_file_content(main_js_path)
    if not main_js_content:
        print_warning("`src/main.js` not found. Components will be non-interactive.")

    all_pages_data = {}
    
    # 4. Process Shared Components (Header/Footer)
    print_info("\nStep 4: Processing shared layout components (Header/Footer)...")
    shared_components_dir = os.path.join(existing_project_path, "src", "components")
    for comp_name in ["header", "footer"]:
        comp_html_path = os.path.join(shared_components_dir, f"{comp_name}.html")
        comp_html_content = read_file_content(comp_html_path)
        if comp_html_content:
            component_name_pascal = to_pascal_case(comp_name)
            # Header/Footer can also have interactivity
            tsx_content, _ = convert_html_to_component(model, comp_html_content, component_name_pascal, main_js_content)
            write_to_file(os.path.join(new_project_path, "src", "components", "layout", f"{component_name_pascal}.tsx"), tsx_content)

    # 5. Process Pages into Sections
    print_info("\nStep 5: Discovering and processing all pages...")
    # FIX: Use recursive glob to find all HTML files
    page_files = glob.glob(os.path.join(existing_project_path, "**", "*.html"), recursive=True)
    # Exclude shared components from the page list
    page_files = [p for p in page_files if shared_components_dir not in os.path.dirname(p)]
    
    for page_path in page_files:
        page_name_raw = os.path.splitext(os.path.basename(page_path))[0]
        slug = "" if page_name_raw.lower() in ['index', 'home'] else page_name_raw
        print_info(f"\n--- Processing Page for slug: `/{slug}` ---")
        
        html_content = read_file_content(page_path)
        if not html_content: continue
        body_match = re.search(r"<body.*?>(.*)</body>", html_content, re.DOTALL)
        if not body_match: continue
        
        sections_data = identify_and_split_sections_ai(model, body_match.group(1).strip())
        
        page_data_entry = []
        for section in sections_data:
            component_name = section['name']
            section_html = section['html']
            
            tsx_content, json_data_str = convert_html_to_component(model, section_html, component_name, main_js_content)
            
            # Write the component file to the correct location
            write_to_file(os.path.join(new_project_path, "src", "components", "sections", f"{component_name}.tsx"), tsx_content)
            
            try:
                props_data = json.loads(json_data_str)
                page_data_entry.append({"component": component_name, "props": props_data})
            except json.JSONDecodeError:
                page_data_entry.append({"component": component_name, "props": {}})

        all_pages_data[slug] = page_data_entry

    # 6. Create Central Data File and Dynamic Page Route
    # The logic here is solid and remains unchanged. It correctly builds the data map and the dynamic page.
    print_info("\nStep 6: Assembling the data-driven routing system...")
    all_component_names = {entry['component'] for slug_data in all_pages_data.values() for entry in slug_data}
    imports = "\n".join([f"import {name} from '@/components/sections/{name}';" for name in all_component_names])
    component_map = f"export const componentMap: {{ [key: string]: React.ComponentType<any> }} = {{\n  " + \
                    ",\n  ".join([f'"{name}": {name}' for name in all_component_names]) + "\n};"
    page_data_ts = f"""// Auto-generated by conversion script
{imports}
interface PageComponent {{ component: string; props: any; }}
export const pages: {{ [key: string]: PageComponent[] }} = {json.dumps(all_pages_data, indent=2)};
{component_map}
"""
    write_to_file(os.path.join(new_project_path, "src", "lib", "pageData.ts"), page_data_ts)
    
    slug_page_tsx = """
import { pages, componentMap } from '@/lib/pageData';
import { notFound } from 'next/navigation';

export default async function Page({ params }: { params: { slug?: string[] } }) {
  const slug = (params.slug || []).join('/') || '';
  const pageData = pages[slug];
  if (!pageData) notFound();
  return (
    <main>
      {pageData.map((section, index) => {
        const Component = componentMap[section.component];
        if (!Component) return null;
        return <Component key={index} {...section.props} />;
      })}
    </main>
  );
}
export async function generateStaticParams() {
  return Object.keys(pages).map((slug) => ({ slug: slug ? slug.split('/') : [] }));
}
"""
    write_to_file(os.path.join(new_project_path, "src", "app", "[[...slug]]", "page.tsx"), slug_page_tsx)
    default_page = os.path.join(new_project_path, 'src', 'app', 'page.tsx')
    if os.path.exists(default_page): os.remove(default_page)

    # 7. Finalize Layout and Assets
    print_info("\nStep 7: Finalizing layout and copying assets...")
    layout_path = os.path.join(new_project_path, "src", "app", "layout.tsx")
    layout_content = read_file_content(layout_path)
    if layout_content:
        layout_content = layout_content.replace(
            "import './globals.css';",
            "import './globals.css';\nimport Header from '@/components/layout/Header';\nimport Footer from '@/components/layout/Footer';"
        )
        layout_content = re.sub(
            r"<body>(.*?)</body>",
            r"<body><Header />{'{children}'}<Footer /></body>",
            layout_content, flags=re.DOTALL
        )
        write_to_file(layout_path, layout_content)
    
    # FIX: Robustly find and copy the public/img folder
    source_img_path = os.path.join(existing_project_path,"src", "public", "img")
    dest_public_path = os.path.join(new_project_path, "public")
    if os.path.exists(source_img_path):
        shutil.copytree(source_img_path, os.path.join(dest_public_path, "img"), dirs_exist_ok=True)
        print_success("Copied `public/img` folder successfully.")
    else:
        print_warning("`public/img` folder not found in existing project root. Skipping.")

    print_success("\n\n--- INTELLIGENT CONVERSION COMPLETE! ---")

if __name__ == "__main__":
    main()