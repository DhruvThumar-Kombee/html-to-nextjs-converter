import os
import json
import pathlib
import subprocess
from dotenv import load_dotenv

# --- LangChain Core Imports ---
from langchain.agents import AgentExecutor, create_json_chat_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool, Tool

# --- LangChain AI Model and Embeddings Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# --- LangChain RAG (Vector Store) Imports ---
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter


# #############################################################################
# SECTION 1: TOOL DEFINITIONS
# #############################################################################

@tool
def read_file(file_path: str) -> str:
    """Reads the entire content of a specified file relative to the project root."""
    project_path = os.getenv("PROJECT_PATH")
    try:
        absolute_path = os.path.join(project_path, file_path)
        with open(absolute_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file {file_path}: {e}"

@tool
def write_file(file_path: str, content: str) -> str:
    """Writes content to a specified file relative to the project root."""
    project_path = os.getenv("PROJECT_PATH")
    try:
        absolute_path = os.path.join(project_path, file_path)
        os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
        with open(absolute_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {absolute_path}."
    except Exception as e:
        return f"Error writing to file {file_path}: {e}"

@tool
def list_directory(path: str) -> str:
    """Lists all files and directories in a specified path relative to the project root."""
    project_path = os.getenv("PROJECT_PATH")
    try:
        absolute_path = os.path.join(project_path, path)
        files = os.listdir(absolute_path)
        return "\n".join(files)
    except Exception as e:
        return f"Error listing directory {path}: {e}"

@tool
def run_shell_command(command: str) -> str:
    """
    Executes a shell command in the project's root directory.
    CRITICAL: You MUST use this to install dependencies or run validation checks.
    Example commands: 'npm install swiper', 'npx tsc --noEmit'
    """
    project_path = os.getenv("PROJECT_PATH")
    if not project_path:
        return "Error: PROJECT_PATH is not set in the .env file."

    try:
        process = subprocess.run(
            command,
            cwd=project_path,
            check=True,
            capture_output=True,
            text=True,
            shell=True
        )
        return f"Command executed successfully. Output:\n{process.stdout}"
    except subprocess.CalledProcessError as e:
        return f"Command failed with exit code {e.returncode}. Stderr:\n{e.stderr}\nStdout:\n{e.stdout}"

@tool
def validate_typescript_code() -> str:
    """
    Runs the TypeScript compiler to check for type errors in the entire project.
    You MUST run this tool after writing or modifying any .ts or .tsx file to ensure your changes are valid.
    An successful empty output means no errors were found.
    """
    return run_shell_command("npx tsc --noEmit")


# #############################################################################
# SECTION 2: RAG SYSTEM
# #############################################################################

class CodeIndexer:
    """Handles indexing and retrieval of the project's codebase for RAG."""
    def __init__(self, project_path: str, embeddings_model):
        self.project_path = pathlib.Path(project_path)
        self.vector_store_path = self.project_path.parent / ".agent_vector_store"
        self.embeddings = embeddings_model
        self.vector_store = Chroma(
            persist_directory=str(self.vector_store_path),
            embedding_function=self.embeddings,
        )

    def index_project(self):
        """Indexes all .ts and .tsx files in the project, excluding node_modules."""
        print(f"🔍 Indexing project files from: {self.project_path}...")
        
        loader = DirectoryLoader(
            str(self.project_path),
            glob="**/*.{ts,tsx}",
            loader_cls=TextLoader,
            show_progress=True,
            use_multithreading=True,
            silent_errors=True
        )
        all_documents = loader.load()
        
        documents = [
            doc for doc in all_documents 
            if 'node_modules' not in pathlib.Path(doc.metadata['source']).parts
        ]

        if not documents:
            print("⚠️ No TypeScript documents found to index. Please ensure your PROJECT_PATH is correct and contains .ts/.tsx files.")
            return

        ts_splitter = RecursiveCharacterTextSplitter.from_language(
            language="typescript", chunk_size=1000, chunk_overlap=100
        )
        chunks = ts_splitter.split_documents(documents)

        print(f"   - Found {len(documents)} files, split into {len(chunks)} chunks.")
        print(f"   - Storing embeddings in: {self.vector_store_path}")

        self.vector_store = Chroma.from_documents(
            chunks, 
            self.embeddings, 
            persist_directory=str(self.vector_store_path)
        )
        self.vector_store.persist()
        print("✅ Project indexing complete.")

    def get_retriever(self):
        """Returns a retriever object for the indexed vector store."""
        return self.vector_store.as_retriever(search_kwargs={"k": 5})


# #############################################################################
# SECTION 3: AGENT CONFIGURATION (NEW AUTONOMOUS PLAN)
# #############################################################################

AGENT_TEMPLATE = """
You are an expert AI software engineer specializing in Next.js and Contentstack integration.
. Your mission is to refactor a static Next.js project to fetch its content from Contentstack.

**CORE REFACTORING PRINCIPLES:**
- **Preserve Existing Styles:** Your primary goal is to make components dynamic, NOT to restyle them. You **MUST** preserve all existing Tailwind CSS class names from the original static components. The new libraries you add are for LOGIC ONLY (state, transitions, etc.).
- **Methodical & Validated:** You follow the plan step-by-step. You never make assumptions. You **ALWAYS** run `validate_typescript_code` after writing or changing any TS/TSX file to ensure correctness.

**THE PLAN (Follow these steps strictly):**

1.  **Initial Analysis:**
    - Read the Contentstack `schema.json` to understand the data structures.
    - Read `pages-components-list.txt` to understand the page-to-component mapping.
    - List the files in `src/components/sections` to know which components need refactoring.

2.  **Core Setup:**
    - Create `.env.local` using the provided snippet.
    - Create `src/lib/contentstackClient.ts` using the provided snippet.
    - Run `npm install @contentstack/delivery-sdk @contentstack/live-preview-utils`.

3.  **ESLint Configuration:**
    - Find and read the project's ESLint config file.
    - Modify the configuration to add the rule `rules: {{ "@typescript-eslint/no-explicit-any": "off" }}`.

4.  **Type Generation (with Self-Correction):**
    - Generate TypeScript types from the `schema.json` and write them to `src/types/contentstack.d.ts`.
    - **VALIDATE:** Run `validate_typescript_code` and fix any errors.

5.  **Query Generation (with Self-Correction):**
    - Generate typed query files in `src/queries` for the header, footer, and pages using the provided snippets and structures.
    - **VALIDATE:** Run `validate_typescript_code` and fix any errors.

6.  **Intelligent Component Refactoring (Loop through each component):**
    - For EACH component in `src/components/sections`:
        a. **Analyze Component:** Read the component's source file. Analyze its JSX structure to determine its functionality. (e.g., Is it a list of items that could be a carousel? Is it a set of questions and answers that should be an accordion?).
        b. **Select & Install Library:** Based on your analysis, choose a suitable, popular, and well-maintained library. Prefer "headless" libraries that don't impose their own styles (e.g., `Swiper.js` for carousels, `Radix UI` for accordions/dialogs). Use `run_shell_command` to install the library (e.g., `npm install swiper`). If no special library is needed, skip this.
        c. **Refactor & Integrate:** Rewrite the component to:
           i. Accept a `data` prop with the types you generated.
           ii. Use the LOGIC from the library you just installed.
           iii. **CRITICAL:** Map the dynamic data from the `data` prop onto the ORIGINAL JSX elements, keeping the **EXACT SAME Tailwind CSS classes**.
        d. **Write Changes:** Save the new, refactored code back to the file.
        e. **VALIDATE:** Run `validate_typescript_code` and fix any errors you introduced.

7.  **Final Assembly & Validation:**
    - Refactor `src/app/layout.tsx` to fetch global data (header/footer).
    - Refactor the dynamic page `src/app/**/[[...slug]]/page.tsx` using the specific, detailed prompt provided for page rendering logic.
    - Address common issues like Contentstack field name mismatches (`page_components` vs `page_components_primary`) or missing image hostnames in `next.config.ts` as needed.
    - **FINAL VALIDATION:** Run `validate_typescript_code` one last time to ensure the entire project is error-free.

**TOOLS:**
{tools}

To use a tool, respond with a JSON blob containing 'action' and 'action_input'. The 'action' must be one of: {tool_names}
**FINAL INSTRUCTION:**
Once you have completed all steps in the plan and the final validation passes, you MUST respond with your final answer in a JSON block. Your final answer should be a summary of the work completed. For example:
```json
{{
  "action": "final_answer",
  "action_input": "I have successfully completed the refactoring of the Next.js project. All components have been made dynamic, necessary libraries have been installed, and the project passes all type-checking validations. The mission is accomplished."
}}
```

**YOUR TASK:**
Begin your mission. Here is the initial input with necessary code snippets and context:
{input}

{agent_scratchpad}
"""

# #############################################################################
# SECTION 4: MAIN EXECUTION LOGIC
# #############################################################################

def main():
    """Main function to configure and run the AI agent."""
    load_dotenv()
    project_path_str = os.getenv("PROJECT_PATH")
    schema_path_str = os.getenv("SCHEMA_JSON_PATH")

    if not all([project_path_str, os.getenv("GEMINI_API_KEY")]):
        print("❌ Error: Please set PROJECT_PATH and GEMINI_API_KEY in your .env file.")
        return

    project_path = pathlib.Path(project_path_str)
    schema_path = pathlib.Path(schema_path_str)

    print("--- 🚀 Starting AI Refactoring Agent ---")

    google_api_key = os.getenv("GEMINI_API_KEY")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro", 
        temperature=0.0, 
        google_api_key=google_api_key,
        max_retries=6,
        timeout=300.0, # Increased to 5 minutes to prevent timeout
        convert_system_message_to_human=True
    )
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=google_api_key,
        max_retries=3
    )

    indexer = CodeIndexer(project_path_str, embeddings)
    indexer.index_project()
    retriever = indexer.get_retriever()

    rag_tool = Tool(
        name="CodebaseRetriever",
        func=lambda query: retriever.get_relevant_documents(query),
        description="Retrieves relevant code snippets, types, and context from the project codebase."
    )

    tools = [read_file, write_file, list_directory, run_shell_command, validate_typescript_code, rag_tool]

    prompt = PromptTemplate.from_template(AGENT_TEMPLATE)

    agent = create_json_chat_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=15000
    )
    
    # --- TEMPLATES FOR AGENT INSTRUCTIONS ---
    # These snippets remain the same as they are for the core setup
    CONTENTSTACK_CLIENT_TS_CODE_SNIPPET = """import contentstack, { Region, StackConfig } from '@contentstack/delivery-sdk';
import ContentstackLivePreview, { IStackSdk } from '@contentstack/live-preview-utils';

const livePreviewConfig = {
  enable: process.env.NEXT_PUBLIC_CONTENTSTACK_PREVIEW === 'true',
  preview_token: process.env.NEXT_PUBLIC_CONTENTSTACK_PREVIEW_TOKEN,
  host:
    process.env.NEXT_PUBLIC_CONTENTSTACK_PREVIEW === 'true'
      ? process.env.NEXT_PUBLIC_CONTENTSTACK_PREVIEW_REST_HOST
      : process.env.NEXT_PUBLIC_CONTENTSTACK_REST_HOST
};

const stackConfig: StackConfig = {
  apiKey: process.env.NEXT_PUBLIC_CONTENTSTACK_API_KEY as string,
  deliveryToken: process.env.NEXT_PUBLIC_CONTENTSTACK_DELIVERY_TOKEN as string,
  environment: process.env.NEXT_PUBLIC_CONTENTSTACK_ENVIRONMENT as string,
  region: Region[process.env.NEXT_PUBLIC_CONTENTSTACK_REGION?.toUpperCase() as keyof typeof Region] || Region.US,
  live_preview: livePreviewConfig,
  branch: process.env.NEXT_PUBLIC_CONTENTSTACK_BRANCH || 'main'
};

export const stack = contentstack.stack(stackConfig);

export function initLivePreview() {
  if (typeof window !== 'undefined') {
    ContentstackLivePreview.init({
      ssr: false,
      enable: process.env.NEXT_PUBLIC_CONTENTSTACK_PREVIEW === 'true',
      mode: 'builder',
      stackSdk: stack.config as IStackSdk,
      stackDetails: {
        apiKey: process.env.NEXT_PUBLIC_CONTENTSTACK_API_KEY as string,
        environment: process.env.NEXT_PUBLIC_CONTENTSTACK_ENVIRONMENT as string
      },
      clientUrlParams: {
        host: process.env.NEXT_PUBLIC_CONTENTSTACK_HOST
      },
      editButton: {
        enable: process.env.NEXT_PUBLIC_CONTENTSTACK_LIVE_EDIT_TAGS === 'true',
      }
    });
  }
}""" 
    GET_HEADER_QUERY_SNIPPET = """import { Header } from '@/types/contentstack';
import { stack } from '@/lib/contentstackClient';
import contentstack from '@contentstack/delivery-sdk';

export const getHeader = async (): Promise<Header | null> => {
  try {
    const result: any = await stack
      .contentType('header')
      .entry()
      .query()
      .addParams({ include_all: true, include_all_depth: 5 })
      .find<Header>();

 

    if (!result || !result.entries || result.entries.length === 0) {
      return null;
    }
    
    const entry = result.entries[0];
    contentstack.Utils.addEditableTags(entry, 'header', true);
    return entry;
  } catch (error) {
    console.error('Error fetching header:', error);
    return null;
  }
};""" 
    GET_FOOTER_QUERY_SNIPPET = """import { Footer } from '@/types/contentstack';
import { stack } from '@/lib/contentstackClient';
import contentstack from '@contentstack/delivery-sdk';

export const getFooter = async (): Promise<Footer | null> => {
  try {
    const result: any = await stack
      .contentType('footer')
      .entry()
      .query()
      .addParams({ include_all: true, include_all_depth: 5 })
      .find<Footer>();



    const entry = result.entries[0];
    if (!result || !result.entries || result.entries.length === 0) {
      return null;
    }
    
    contentstack.Utils.addEditableTags(entry, 'footer', true, 'en-us');
    return entry;
  } catch (error) {
    console.error('Error fetching footer:', error);
    return null;
  }
};"""  # (Code from your script, kept for brevity)
    ENV_LOCAL_CONTENT = """# Contentstack Credentials - Fill these with your stack details
NEXT_PUBLIC_CONTENTSTACK_API_KEY=""
NEXT_PUBLIC_CONTENTSTACK_DELIVERY_TOKEN=""
NEXT_PUBLIC_CONTENTSTACK_ENVIRONMENT=""
NEXT_PUBLIC_CONTENTSTACK_BRANCH="main"
# Contentstack Preview & Live Edit Configuration
NEXT_PUBLIC_CONTENTSTACK_PREVIEW="false"
NEXT_PUBLIC_CONTENTSTACK_PREVIEW_TOKEN=""
NEXT_PUBLIC_CONTENTSTACK_LIVE_EDIT_TAGS="true"
# Host & Region Configuration (change if you are not in the US region)
NEXT_PUBLIC_CONTENTSTACK_HOST="app.contentstack.com"
NEXT_PUBLIC_CONTENTSTACK_REGION="us"
NEXT_PUBLIC_CONTENTSTACK_REST_HOST="cdn.contentstack.io"
NEXT_PUBLIC_CONTENTSTACK_PREVIEW_REST_HOST="rest-preview.contentstack.com"
"""
    PAGE_TSX_REFACTOR_PROMPT = """You are now in the page refactoring phase. Your task is to generate the complete code for the dynamic page file located at `src/app/**/[[...slug]]/page.tsx`.

**CRITICAL INSTRUCTIONS:**
- Use the provided `pages-components-list.txt` mapping as the absolute source of truth for which component to render for each page type and block.
- Follow the exact structure outlined below. Do not add extra features.
- Ensure all necessary components are imported from `../../components/sections/`.
- Use `(block as any)` for type assertion where necessary to access modular block properties dynamically.

**File Structure to Generate:**

1.  **Imports**:
    - Import `notFound` from 'next/navigation'.
    - Import `getPage` from '@/queries/getPage'.
    - Import all required types from '@/types/contentstack'.
    - Import ALL relevant section components from `../../components/sections/`.

2.  **Page Function**:
    - Signature: `export default async function Page({ params }: { params: { slug?: string[] } })`.
    - Logic: Construct the `url` from the `slug` array. Fetch the `page` data using `getPage(url)`. If `!page`, call `notFound()`.

3.  **`renderComponent` Function**:
    - Define this helper function inside the `Page` component.
    - It must accept `block` and `index` as arguments.
    - It must determine the `pageType` from the URL (e.g., `'/'` becomes `'home'`, `'/about-us'` becomes `'about-us'`).
    - Use a `switch (pageType)` or `if-else` chain to handle rendering for each page defined in the mapping.
    - Inside each page's logic, use `if ('hero_section' in block)` to check the block type from the Contentstack response.
    - Render the correct component, passing `key={index}` and `data={(block as any).hero_section.section_reference[0]}`. Note the dynamic access and `.section_reference[0]` pattern.

4.  **Return Statement**:
    - Return a `<main>` element.
    - Map over `page?.page_components?.map((block, index) => renderComponent(block, index))` .

*Contentstack Field Name Mismatch**: If page data exists but components don't render:
   - Check if using `page_components` instead of `page_components_primary`
   - Update: `page?.page_components_primary?.map()` instead of `page?.page_components?.map()`
   - Verify TypeScript types include both fields: `page_components_primary?: ModularBlock[]`
2. **Image Hostname Error**: If you see "hostname is not configured under images" error:
   - Add image configuration to next.config.ts:
   ```typescript
   const nextConfig: NextConfig = {
     images: {
       remotePatterns: [
         {
           protocol: 'https',
           hostname: 'images.contentstack.io',
           port: '',
           pathname: '/**',
         },
       ],
     },
   };

Now, using the original file content and the mapping as context, generate the complete, refactore""" 
    
    # NOTE: To run this, you must copy the full string content for the variables above
    # from your original script. I've truncated them here to focus on the changes.
    
    try:
        schema_content = schema_path.read_text(encoding='utf-8')
        mapping_file_path = project_path.parent / "pages-components-list.txt"
        component_mapping_content = mapping_file_path.read_text(encoding='utf-8')

        initial_input = f"""
        Begin the Contentstack integration process for the Next.js project located at `{project_path}`.

        **CRITICAL INSTRUCTION: Core File Snippets**
        When you reach the setup steps, you MUST use the following code snippets exactly as provided for the corresponding files.

        **File: `.env.local`**
        ```
        {ENV_LOCAL_CONTENT}
        ```

        **File: `src/lib/contentstackClient.ts`**
        ```typescript
        {CONTENTSTACK_CLIENT_TS_CODE_SNIPPET}
        ```

        **File: `getHeader.ts`**
        ```typescript
        {GET_HEADER_QUERY_SNIPPET}
        ```

        **File: `getFooter.ts`**
        ```typescript
        {GET_FOOTER_QUERY_SNIPPET}
        ```
        
        **Structure for `getPage.ts`:**
        - Discover the page content type UID from the schema (look for 'url' and 'modular_blocks' fields).
        - Use this structure: `const result = await stack.contentType('YOUR_UID_HERE').entry().query().where('url', QueryOperation.EQUALS, url).find<YOUR_UID_HERE>()
        - Add editable tags using the discovered UID.

        **CRITICAL INSTRUCTION: Dynamic Page (`[[...slug]]/page.tsx`) Refactoring**
        When you refactor the dynamic page, you MUST follow these detailed instructions:
        ---
        {PAGE_TSX_REFACTOR_PROMPT}
        ---

        **ADDITIONAL CONTEXT:**
        The Contentstack schema is:
        --- SCHEMA START ---
        {schema_content}
        --- SCHEMA END ---

        The page-to-component mapping is:
        --- MAPPING START ---
        {component_mapping_content}
        --- MAPPING END ---
        """

        print("\n--- 🤖 Invoking Agent (This will take a while, and will auto-retry on server errors) ---\n")
        result = agent_executor.invoke({
            "input": initial_input,
            "rag_context": ""
        })
        
        print("\n--- ✅ Agent execution finished! ---")
        print(f"Final Output: {result.get('output')}")

    except FileNotFoundError as e:
        print(f"❌ FATAL ERROR: A required file was not found. Please check your paths in the .env file and ensure pages-components-list.txt exists.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during agent execution: {e}")

if __name__ == "__main__":
    main()
