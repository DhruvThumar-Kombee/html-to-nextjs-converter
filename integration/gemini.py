import os
import json
import pathlib
import time
import subprocess
from dotenv import load_dotenv
import google.generativeai as genai

# --- HELPER FUNCTION TO RUN COMMANDS ---
def run_command(command, cwd, step_description):
    """Runs a shell command in a specified directory and prints its status."""
    print(f"   - {step_description}...")
    try:
        process = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            shell=True if os.name == 'nt' else False # Use shell=True on Windows for npm
        )
        print(f"     - SUCCESS: {step_description} completed.")
    except subprocess.CalledProcessError as e:
        print(f"     - FAILED: {step_description}.")
        print(f"     - Error: {e.stderr}")
        raise e

# --- INTELLIGENT SCHEMA ANALYSIS ---
def find_page_content_type_uid(schema_data):
    """
    Analyzes the schema to find the UID of the main 'page' content type.
    It looks for a content type that has both a 'url' field and a 'modular_blocks' field.
    """
    print("   - Analyzing schema to find page content type UID...")
    # --- FIX IS HERE ---
    # The root of the schema export is a list, so we iterate directly over it.
    for ct in schema_data:
        has_url_field = False
        has_modular_blocks = False
        # Each 'ct' is a dictionary representing a content type
        for field in ct.get('schema', []):
            if field.get('uid') == 'url' and field.get('data_type') == 'text':
                has_url_field = True
            if field.get('data_type') == 'modular_blocks':
                has_modular_blocks = True
        if has_url_field and has_modular_blocks:
            print(f"     - Found page content type: '{ct['uid']}'")
            return ct['uid']
    print("     - WARNING: Could not auto-detect page content type. Falling back to 'page'.")
    return 'page' # Fallback UID


# --- PROVIDED CODE SNIPPETS (UNCHANGED) ---
CONTENTSTACK_CLIENT_TS_CODE_SNIPPET = """
import contentstack, { Region, StackConfig } from '@contentstack/delivery-sdk';
import ContentstackLivePreview, { IStackSdk } from '@contentstack/live-preview-utils';

const livePreviewConfig = {
  enable: process.env.NEXT_PUBLIC_CONTENTSTACK_PREVIEW === 'true',
  preview_token: process.env.NEXT_PUBLIC_CONTENTSTACK_PREVIEW_TOKEN,
  host:
    process.env.NEXT_PUBLIC_CONTENTSTACK_PREVIEW === 'true'
      ? process.env.NEXT_PUBLIC_CONTENTSTACK_PREVIEW_REST_HOST // Should be region specific preview host
      : process.env.NEXT_PUBLIC_CONTENTSTACK_REST_HOST     // Should be region specific delivery host (cdn)
};

const stackConfig: StackConfig = {
  apiKey: process.env.NEXT_PUBLIC_CONTENTSTACK_API_KEY as string,
  deliveryToken: process.env.NEXT_PUBLIC_CONTENTSTACK_DELIVERY_TOKEN as string,
  environment: process.env.NEXT_PUBLIC_CONTENTSTACK_ENVIRONMENT as string,
  region: (process.env.NEXT_PUBLIC_CONTENTSTACK_REGION?.toUpperCase() as keyof typeof Region) || Region.US,
  live_preview: livePreviewConfig,
  branch: process.env.NEXT_PUBLIC_CONTENTSTACK_BRANCH || 'main'
};

export const stack = contentstack.stack(stackConfig);

export function initLivePreview() {
  if (typeof window !== 'undefined') { // Ensure this runs only on client
    ContentstackLivePreview.init({
      ssr: false,
      enable: process.env.NEXT_PUBLIC_CONTENTSTACK_PREVIEW === 'true',
      mode: 'builder',
      stackSdk: stack.config as IStackSdk, // Cast to IStackSdk
      stackDetails: {
        apiKey: process.env.NEXT_PUBLIC_CONTENTSTACK_API_KEY as string,
        environment: process.env.NEXT_PUBLIC_CONTENTSTACK_ENVIRONMENT as string
      },
      clientUrlParams: {
        host: process.env.NEXT_PUBLIC_CONTENTSTACK_HOST // e.g., app.contentstack.com or eu-app.contentstack.com
      },
      editButton: {
        enable: process.env.NEXT_PUBLIC_CONTENTSTACK_LIVE_EDIT_TAGS === 'true',
      }
    });
  }
}
"""

LIVE_PREVIEW_INIT_TSX_CODE_SNIPPET = """
'use client';
import { useEffect } from 'react';
import { initLivePreview } from '@/lib/contentstackClient';
export default function LivePreviewInit() {
  useEffect(() => {
    if (process.env.NEXT_PUBLIC_CONTENTSTACK_PREVIEW === 'true') {
      initLivePreview();
    }
  }, []);
  return null;
}
"""

NEXT_CONFIG_TS_CODE_SNIPPET = """
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    remotePatterns: [
     { protocol: 'https', hostname: 'placehold.co' },
     { protocol: 'https', hostname: 'via.placeholder.com' },
     { protocol: 'https', hostname: 'images.contentstack.io' },
    ],
    dangerouslyAllowSVG: true,
  },
};
export default nextConfig;
"""

ENV_LOCAL_CONTENT = """
# Contentstack Credentials - Fill these with your stack details
NEXT_PUBLIC_CONTENTSTACK_API_KEY=""
NEXT_PUBLIC_CONTENTSTACK_DELIVERY_TOKEN=""
NEXT_PUBLIC_CONTENTSTACK_ENVIRONMENT=""
NEXT_PUBLIC_CONTENTSTACK_BRANCH="main"
# Contentstack Preview & Live Edit Configuration
NEXT_PUBLIC_CONTENTSTACK_PREVIEW="true"
NEXT_PUBLIC_CONTENTSTACK_PREVIEW_TOKEN=""
NEXT_PUBLIC_CONTENTSTACK_LIVE_EDIT_TAGS="true"
# Host & Region Configuration (change if you are not in the US region)
NEXT_PUBLIC_CONTENTSTACK_HOST="app.contentstack.com"
NEXT_PUBLIC_CONTENTSTACK_REGION="us"
# These hosts are derived from your region.
NEXT_PUBLIC_CONTENTSTACK_PREVIEW_REST_HOST="rest-preview.contentstack.com"
NEXT_PUBLIC_CONTENTSTACK_REST_HOST="cdn.contentstack.io"
"""

# --- SCRIPT CONFIGURATION AND HELPER FUNCTIONS ---

def configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-pro-preview-05-06')

def clean_gemini_response(response_text):
    lines = response_text.strip().split('\n')
    if lines and (lines[0].strip().startswith('```') or lines[0].strip().startswith('```typescript')):
        lines = lines[1:]
    if lines and lines[-1].strip() == '```':
        lines = lines[:-1]
    return '\n'.join(lines)

def validate_generated_code(code, file_type):
    """Validate generated code for common issues."""
    if not code:
        return False
    
    # Check for common issues
    issues = []
    
    if file_type == "query":
        if "result.entries[0]" in code and "const entry = result.entries[0];" not in code:
            issues.append("Missing entry extraction")
        if "addEditableTags" in code and "entry" not in code:
            issues.append("Using addEditableTags without proper entry extraction")
        
    
    if file_type == "component":
        if "data?" not in code and "data." in code:
            issues.append("Missing optional chaining")
        if "Image" in code and "width" not in code:
            issues.append("Missing required Image props")
        if "data?.image?.url" in code and "data?.image?.url &&" not in code:
            issues.append("Missing conditional rendering for images")
    
    if issues:
        print(f"    - WARNING: Generated code has issues: {', '.join(issues)}")
        return False
    
    return True

def make_api_call_with_retry(model, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            print(f"    - Making Gemini API call (Attempt {attempt + 1}/{max_retries})...")
            response = model.generate_content(prompt)
            if not response.parts:
                 print("    - Received an empty response from API. Retrying...")
                 time.sleep(5)
                 continue
            return clean_gemini_response(response.text)
        except Exception as e:
            print(f"    - API call failed: {e}")
            if "API key" in str(e):
                raise e
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                print("    - Max retries reached. Skipping this file.")
                return None

# --- SCRIPT LOGIC ---

def setup_project_structure(project_path):
    print("1. Setting up directory structure...")
    dirs_to_create = [
        project_path / "src" / "lib",
        project_path / "src" / "components" / "sections",
        project_path / "src" / "queries",
        project_path / "src" / "types"
    ]
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
    print("   - Directory setup complete.\n")

def create_core_files(project_path):
    print("2. Creating core Contentstack integration files...")
    files_to_create = {
        "src/lib/contentstackClient.ts": CONTENTSTACK_CLIENT_TS_CODE_SNIPPET,
        "src/components/LivePreviewInit.tsx": LIVE_PREVIEW_INIT_TSX_CODE_SNIPPET,
        "next.config.mjs": NEXT_CONFIG_TS_CODE_SNIPPET,
        ".env.local": ENV_LOCAL_CONTENT
    }
    js_config_path = pathlib.Path(project_path) / "next.config.js"
    if js_config_path.exists():
        js_config_path.unlink()
    for rel_path, content in files_to_create.items():
        file_path = pathlib.Path(project_path) / rel_path
        file_path.write_text(content.strip(), encoding='utf-8')
    print("   - Core file creation complete.\n")

def generate_typescript_types(project_path, schema_content, model):
    print("3. Generating TypeScript types from schema...")
    types_file_path = project_path / "src" / "types" / "contentstack.d.ts"
    prompt = f"""
    You are an expert TypeScript developer. Your task is to generate TypeScript interfaces based on a Contentstack JSON schema export.
    **Instructions:**
    1.  Analyze the provided JSON schema. For each content type and each modular block, generate a corresponding `export interface` or `export type`. Use PascalCase for names (e.g., `hero_section` becomes `HeroSection`).
    2.  For any `File` fields, use the `Image` type. You MUST include this import at the top: `import {{ Image }} from '@contentstack/delivery-sdk';`.
    3.  The final output must be ONLY the raw TypeScript code.
    **Contentstack Schema JSON:**
    ```json\n{schema_content}\n```
    """
    generated_types = make_api_call_with_retry(model, prompt)
    if generated_types:
        types_file_path.write_text(generated_types.strip(), encoding='utf-8')
        print(f"   - SUCCESS: TypeScript types generated at {types_file_path}")
        return generated_types
    else:
        print(f"   - FAILED: Could not generate TypeScript types.")
        return None

def generate_query_files(project_path, page_content_type_uid, model):
    print("4. Generating typed query files...")
    queries_dir = project_path / "src" / "queries"
    
    # Dynamically create the getPage query snippet with the detected UID
    get_page_query_dynamic_snippet = f"""
import contentstack, {{ QueryOperation }} from '@contentstack/delivery-sdk';
import {{ stack }} from '@/lib/contentstackClient';
import {{ DynamicPage }} from '@/types/contentstack';

export const getPage = async (url: string): Promise<DynamicPage | null> => {{
  try {{
    console.log('🔍 Fetching page for URL:', url);
    console.log('🔧 Environment:', process.env.NEXT_PUBLIC_CONTENTSTACK_ENVIRONMENT);
    
    const result = await stack
      .contentType('{page_content_type_uid}')
      .entry()
      .query()
      .where('url', QueryOperation.EQUALS, url)
      .addParams({{
        include_fallback: true,
       
        include_all: true,
        include_all_depth: 5
      }})
      .find();

    console.log('📊 Query result:', result.entries.length, 'entries found');
    
    if (!result.entries || result.entries.length === 0) {{
      console.warn(`No page found for URL: ${{url}}`);
      return null;
    }}
    
    const entry = result.entries[0]; // ✅ CRITICAL: Extract entry first
    console.log('✅ Page data structure:', Object.keys(entry));
    
    return contentstack.Utils.addEditableTags(entry, '{page_content_type_uid}', entry.uid);

  }} catch (error) {{
    console.error(`Failed to fetch page with URL "${{url}}":`, error);
    return null;
  }}
}};
"""

    query_configs = {
        "Page": {"file_name": "getPage.ts", "code_snippet": get_page_query_dynamic_snippet},
        "Header": {"uid": "header", "function_name": "getHeader", "return_type": "Header | null", "type_name": "Header"},
        "Footer": {"uid": "footer", "function_name": "getFooter", "return_type": "Footer | null", "type_name": "Footer"},
    }
    
    for type_name, config in query_configs.items():
        file_path = queries_dir / f"{config.get('function_name', 'getPage')}.ts"
        generated_code = None
        if 'code_snippet' in config:
            print(f"  - Writing query: {config['file_name']}")
            generated_code = config['code_snippet']
        else:
            print(f"  - Generating AI query for: {config['function_name']}")
            prompt = f"""
            Generate a typed Contentstack query function named `{config['function_name']}`.
            **Requirements:**
            1. It must be `async` and return `Promise<{config['return_type']}>`.
            2. Import `{config['type_name']}` from `@/types/contentstack`.
            3. The query chain must be: `stack.contentType('{config['uid']}').entry().query().find()`.
            4. **CRITICAL:** Extract entry with `const entry = result.entries[0];` before using with `addEditableTags`.
            5. Apply `contentstack.Utils.addEditableTags(entry, '{config['uid']}', entry.uid)`.
            6. Add `addParams({{ include_all: true, include_all_depth: 5 }})` for nested data.
            7. Add console.log statements for debugging: `console.log('🔍 Fetching {config['uid']} data');`
            8. Return the entry, or `null` if not found or on error.
            9. Output ONLY the raw, complete TypeScript code.
            """
            generated_code = make_api_call_with_retry(model, prompt)

        if generated_code:
            # Validate generated code before writing
            if validate_generated_code(generated_code, "query"):
                file_path.write_text(generated_code.strip(), encoding='utf-8')
                print(f"     - SUCCESS: Generated {config.get('function_name', 'getPage')}.ts")
            else:
                print(f"     - WARNING: Generated code has issues, but writing anyway")
                file_path.write_text(generated_code.strip(), encoding='utf-8')
    print("   - Query file generation complete.\n")

def refactor_project_files(project_path, generated_types, model):
    print("5. Starting Intelligent Refactoring Process...")
    app_dir = project_path / "src" / "app"
    files_to_refactor = list(app_dir.rglob("**/page.tsx")) + list(app_dir.rglob("**/layout.tsx"))

    for file_path in files_to_refactor:
        print(f"    - Analyzing: {file_path}")
        try:
            original_content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError as e:
            print(f"    - WARNING: Could not read {file_path.name} due to encoding issues: {e}")
            continue
        prompt = ""

        # Differentiated logic for dynamic slug page vs root page vs layout
        if '[[...slug]]' in str(file_path):
            print("      - Detected dynamic slug page. Applying specific refactoring logic.")
            prompt = f"""
            You are a Next.js v15 expert refactoring a DYNAMIC SLUG page file to use Contentstack data.
            **Target File:** `{file_path.name}` (at `[[...slug]]/page.tsx`)
            **DO NOT CHANGE ANY UI, JSX STRUCTURE, OR CSS CLASSNAMES.**

            **Instructions:**
            1. The component signature must be `export default async function Page({{ params }}: {{ params: {{ slug?: string[] }} }})`.
            2. **Determine the URL from params:** Create the URL path like this: `const url = `/${{params.slug?.join('/') || ''}}`;`.
            3. **Import necessary modules:**
               - `import {{ notFound }} from 'next/navigation';`
               - `import {{ getPage }} from '@/queries/getPage';`
               - Import all required section components from `@/components/sections/`.
               - Import all TypeScript types: `import {{ DynamicPage, PageComponents }} from '@/types/contentstack';`
            4. Fetch data: `const page = await getPage(url);`. Add `if (!page) notFound();`.
            5. **Create this type of logic in a renderComponent helper function for dynamic slug pages:**
            for example:
               ```typescript
               const renderComponent = (block: PageComponents, index: number) => {{
                 console.log('🔍 Rendering block:', block);
                 
                 if ('hero_section_block' in block) {{
                   return <HeroSection key={{index}} data={{block.hero_section_block.section_reference[0]}} />;
                 }}
                 if ('features_grid_block' in block) {{
                   return <FeaturesGrid key={{index}} data={{block.features_grid_block.section_reference[0]}} />;
                
                 return null;
               }};
               ```
            6. **Render sections:** `{{page?.page_components?.map((block, index) => renderComponent(block, index))}}`
            7. Return ONLY the complete, refactored TypeScript code.

            **REFERENCE - Generated Types:**
            ```typescript\n{generated_types}\n```
            **ORIGINAL FILE CONTENT:**
            ```typescript\n{original_content}\n```
            """
        elif file_path.name == 'page.tsx': # This handles the root page.tsx if it exists separately
             print("      - Detected root page. Applying homepage refactoring logic.")
             prompt = f"""
            You are a Next.js v14 expert refactoring a ROOT page file to use Contentstack data.
            **Target File:** `src/app/page.tsx`
            **DO NOT CHANGE ANY UI, JSX STRUCTURE, OR CSS CLASSNAMES.**

            **Instructions:**
            1.  Make the main component `async`.
            2.  **Import necessary modules:**
                - `import {{ getPage }} from '@/queries/getPage';`
                - Import all required section components from `@/components/sections/`.
                - Import all TypeScript types for the sections from `@/types/contentstack`.
            3.  Fetch data for the homepage: `const page = await getPage('/');`.
            4.  Render sections dynamically by mapping over `page?.page_components`.
            5.  Return ONLY the complete, refactored TypeScript code.

            **REFERENCE - Generated Types:**
            ```typescript\n{generated_types}\n```
            **ORIGINAL FILE CONTENT:**
            ```typescript\n{original_content}\n```
             """
        elif file_path.name == 'layout.tsx':
            print("      - Detected layout file. Applying layout refactoring logic.")
            prompt = f"""
            You are a Next.js v15 expert refactoring a LAYOUT file to use Contentstack data.
            **Target File:** `layout.tsx`
            **DO NOT CHANGE ANY UI, JSX STRUCTURE, OR CSS CLASSNAMES.**
            
            **Instructions:**
            1. Make the main component `async`.
            2. Import `getHeader` from `@/queries/getHeader`, `getFooter` from `@/queries/getFooter`, and `LivePreviewInit` from `@/components/LivePreviewInit`.
            3. Fetch header and footer data: `const header = await getHeader();`, `const footer = await getFooter();`.
            4. **Pass data correctly:**
               - For Header: `<Header logo={{header?.logo}} desktopNavLinks={{header?.navigation_links || []}} desktopActionButtons={{header?.action_buttons || []}} mobileNavLinks={{header?.navigation_links || []}} mobileActionButtons={{header?.action_buttons || []}} />`
               - For Footer: `<Footer {{...footer}} />`
            5. Render `<LivePreviewInit />` inside the `<body>` tag.
            6. Add null checks: `if (!header || !footer) return <div>Loading...</div>;`
            7. Return ONLY the complete, refactored TypeScript code.

            **REFERENCE - Generated Types:**
            ```typescript\n{generated_types}\n```
            **ORIGINAL FILE CONTENT:**
            ```typescript\n{original_content}\n```
            """

        if prompt:
            modified_content = make_api_call_with_retry(model, prompt)
            if modified_content:
                # Validate generated code before writing
                if validate_generated_code(modified_content, "page"):
                    file_path.write_text(modified_content, encoding='utf-8')
                    print(f"      - SUCCESS: Refactored {file_path.name}")
                else:
                    print(f"      - WARNING: Generated code has issues, but writing anyway")
                    file_path.write_text(modified_content, encoding='utf-8')
                    print(f"      - SUCCESS: Refactored {file_path.name}")
            else:
                print(f"      - FAILED: Could not refactor {file_path.name}")

    print("\n   - App directory refactoring complete.\n")

    # STAGE 2: Refactor individual section components
    print("  -> STAGE 2: Refactoring individual section components...")
    sections_dir = project_path / "src" / "components" / "sections"
    if not sections_dir.exists():
        print("    - WARNING: `src/components/sections` directory not found. Skipping Stage 2.")
        return
    section_files = list(sections_dir.rglob("*.tsx")) + list(sections_dir.rglob("*.jsx"))
    for file_path in section_files:
        print(f"    - Analyzing: {file_path.name}")
        try:
            original_content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError as e:
            print(f"    - WARNING: Could not read {file_path.name} due to encoding issues: {e}")
            continue
        prompt = f"""
        You are a React/Next.js expert refactoring a component to use Contentstack data and `next/image`.
        **DO NOT CHANGE ANY UI, JSX STRUCTURE, OR CSS CLASSNAMES.**

        **Instructions for `{file_path.name}`:**
        1. **Replace `<img>` tags with `<Image />` from `next/image`**. Import it: `import Image from 'next/image'`.
        2. **DELETE any existing props `interface` or `type` definition** in the file.
        3. **Import the correct TypeScript type** for this section from `@/types/contentstack`. Infer the type name (e.g., for `HeroSection.tsx`, import `HeroSection as HeroSectionType`).
        4. Update the component to accept a single prop: `({{ data }}: {{ data: SectionType }})`.
        5. **Add null checks at the beginning:** `if (!data) return <div>Loading...</div>;`
        6. Update the JSX to access data via the `data` prop using **optional chaining** (e.g., `data?.title`, `data?.image?.url`).
        7. **For `<Image>` components, use conditional rendering:**
           ```typescript
           {{data?.image?.url && (
             <Image
               src={{data.image.url}}
               alt={{data.image.title || 'Default alt text'}}
               width={{data.image.dimension?.width || 500}}
               height={{data.image.dimension?.height || 300}}
               className="responsive-class"
             />
           )}}
           ```
        8. **Handle arrays safely:** `{{(data?.items || []).map((item, index) => (...))}}`
        9. **Add console.log for debugging:** `console.log('🎯 {file_path.name} data:', data);`
        10. Return ONLY the complete, refactored TypeScript code.

        **REFERENCE - Generated Types:**
        ```typescript\n{generated_types}\n```
        **ORIGINAL FILE CONTENT:**
        ```typescript\n{original_content}\n```
        """
        modified_content = make_api_call_with_retry(model, prompt)
        if modified_content:
            # Validate generated code before writing
            if validate_generated_code(modified_content, "component"):
                file_path.write_text(modified_content, encoding='utf-8')
                print(f"      - SUCCESS: Refactored {file_path.name}")
            else:
                print(f"      - WARNING: Generated code has issues, but writing anyway")
                file_path.write_text(modified_content, encoding='utf-8')
                print(f"      - SUCCESS: Refactored {file_path.name}")
        else:
            print(f"      - FAILED: Could not refactor {file_path.name}")
    print("\n   - Project file refactoring complete.\n")


def main():
    """Main function to run the integration script."""
    load_dotenv()
    project_path_str = os.getenv("PROJECT_PATH")
    schema_path_str = os.getenv("SCHEMA_JSON_PATH")

    if not all([project_path_str, schema_path_str]):
        print("Error: Please set PROJECT_PATH and SCHEMA_JSON_PATH in your .env file.")
        return

    project_path = pathlib.Path(project_path_str)
    schema_path = pathlib.Path(schema_path_str)

    if not project_path.is_dir() or not schema_path.is_file():
        print(f"Error: Project or Schema path is invalid. Check your .env file.")
        return

    try:
        print("--- Starting Contentstack Integration Script ---")
        model = configure_gemini()
        schema_content = schema_path.read_text(encoding='utf-8')
        schema_data = json.loads(schema_content)

        setup_project_structure(project_path)
        create_core_files(project_path)
        
        print("2a. Installing Contentstack SDKs...")
        install_sdks_command = ["npm", "install", "@contentstack/delivery-sdk", "@contentstack/live-preview-utils"]
        run_command(install_sdks_command, cwd=project_path, step_description="Install Contentstack SDKs")
        print()

        generated_types = generate_typescript_types(project_path, schema_content, model)
        if not generated_types:
            print("Halting script because TypeScript types could not be generated.")
            return

        # Intelligently find the page UID before generating queries
        page_content_type_uid = find_page_content_type_uid(schema_data)

        generate_query_files(project_path, page_content_type_uid, model)
        refactor_project_files(project_path, generated_types, model)

        # Final validation step
        print("6. Running final validation...")
        queries_dir = project_path / "src" / "queries"
        sections_dir = project_path / "src" / "components" / "sections"
        
        if queries_dir.exists():
            for query_file in queries_dir.glob("*.ts"):
                try:
                    content = query_file.read_text(encoding='utf-8')
                    if not validate_generated_code(content, "query"):
                        print(f"   - WARNING: {query_file.name} may have issues")
                except UnicodeDecodeError as e:
                    print(f"   - WARNING: Could not read {query_file.name} due to encoding issues: {e}")
        
        if sections_dir.exists():
            for component_file in sections_dir.glob("*.tsx"):
                try:
                    content = component_file.read_text(encoding='utf-8')
                    if not validate_generated_code(content, "component"):
                        print(f"   - WARNING: {component_file.name} may have issues")
                except UnicodeDecodeError as e:
                    print(f"   - WARNING: Could not read {component_file.name} due to encoding issues: {e}")

        print("--- Integration Process Complete! ---")
        print("\n**Next Steps:**")
        print(f"1. Open `{project_path / '.env.local'}` and fill in your Contentstack credentials.")
        print("2. **IMPORTANT:** Carefully review the changes made by the script, especially in `src/app/[[...slug]]/page.tsx` and your section components. The AI refactoring is powerful but may require minor manual adjustments.")
        print("3. Start your Next.js development server (`npm run dev`).")

    except Exception as e:
        print(f"\nAn error occurred during the script execution: {e}")

if __name__ == "__main__":
    main()