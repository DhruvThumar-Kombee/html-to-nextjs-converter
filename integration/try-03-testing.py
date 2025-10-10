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
    Example commands: 'npm install @contentstack/delivery-sdk', 'npx tsc --noEmit'
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
# SECTION 3: AGENT CONFIGURATION (CORRECTED VERSION)
# #############################################################################

AGENT_TEMPLATE = """
You are an elite AI software engineer specializing in Next.js and Contentstack integration.
Your sole mission is to refactor a static Next.js project to fetch its content from Contentstack, following a precise plan.

**YOUR PERSONALITY:**
- You are meticulous, precise, and methodical. You follow instructions exactly.
- You never make assumptions. You use your tools to gather information first.
- You are relentless in your pursuit of correctness. You **always** validate your work after making changes.

**THE PLAN (Follow these steps strictly):**

1.  **Setup & Analysis:**
    - Read the Contentstack `schema.json`.
    - Read `pages-components-list.txt`.
    - List the files in `src/components/sections`.

2.  **Core File Creation & Dependencies:**
    - Create `.env.local`, `src/lib/contentstackClient.ts`, and other core files. **YOU MUST USE THE PROVIDED SNIPPETS FOR THIS.**
    - Run `npm install @contentstack/delivery-sdk @contentstack/live-preview-utils`.

3.  **ESLint Configuration:**
    - Read the `eslint.config.js` file (or equivalent).
    - Modify the configuration to add the rule `rules: {{ "@typescript-eslint/no-explicit-any": "off" }}`. This is necessary for pragmatic type assertions during refactoring.

4.  **Type Generation (with Self-Correction):**
    - Generate TypeScript types from the schema and write them to `src/types/contentstack.d.ts`.
    - **VALIDATE:** Run `validate_typescript_code`. Fix any errors.

5.  **Query Generation (with Self-Correction):**
    - Generate typed query files in `src/queries` using the provided snippets and structures.
    - **VALIDATE:** Run `validate_typescript_code`. Fix any errors.

6.  **Component Refactoring (RAG + Self-Correction Loop):**
    - For EACH component in `src/components/sections`:
        a. Refactor the component to be data-driven.
        b. **VALIDATE:** Run `validate_typescript_code` and fix any new errors.

7.  **Page & Layout Refactoring (RAG + Self-Correction Loop):**
    - Refactor `src/app/layout.tsx`.
    - Refactor `src/app/**/[[...slug]]/page.tsx` using the specific, detailed prompt provided for page rendering logic.
    - **VALIDATE:** Run `validate_typescript_code` and fix any final errors.

**TOOLS:**
You have access to the following tools:
{tools}

To use a tool, respond with a JSON blob containing the 'action' and 'action_input' keys.
The 'action' must be one of the following: {tool_names}

**CONTEXT FROM RAG:**
When refactoring, I will provide relevant context retrieved from the codebase. Use this to ensure accuracy.
{rag_context}

**YOUR TASK:**
Begin your mission. Here is the initial input, which includes the up-to-date code snippets and refactoring instructions you must use:
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
        timeout=120.0,
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

    CONTENTSTACK_CLIENT_TS_CODE_SNIPPET = """
import contentstack, { Region, StackConfig } from '@contentstack/delivery-sdk';
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
  region: (process.env.NEXT_PUBLIC_CONTENTSTACK_REGION?.toUpperCase() as keyof typeof Region) || Region.US,
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
}
"""

    GET_HEADER_QUERY_SNIPPET = """
import { Header } from '@/types/contentstack';
import { stack } from '@/lib/contentstackClient';
import contentstack from '@contentstack/delivery-sdk';

export const getHeader = async (): Promise<Header | null> => {
  try {
    const result = await stack
      .contentType('header')
      .entry()
      .query()
      .addParams({ include_references: 2, include_fallback: true })
      .findOne<Header>();

    if (!result) {
      return null;
    }
    
    contentstack.Utils.addEditableTags(result, 'header', true);
    return result;
  } catch (error) {
    console.error('Error fetching header:', error);
    return null;
  }
};
"""

    GET_FOOTER_QUERY_SNIPPET = """
import { Footer } from '@/types/contentstack';
import { stack } from '@/lib/contentstackClient';
import contentstack from '@contentstack/delivery-sdk';

export const getFooter = async (): Promise<Footer | null> => {
  try {
    const result = await stack
      .contentType('footer')
      .entry()
      .query()
      .addParams({ include_references: 2, include_fallback: true })
      .findOne<Footer>();

    if (!result) {
      return null;
    }
    
    contentstack.Utils.addEditableTags(result, 'footer', true);
    return result;
  } catch (error) {
    console.error('Error fetching footer:', error);
    return null;
  }
};
"""
    
    ENV_LOCAL_CONTENT = """
# Contentstack Credentials - Fill these with your stack details
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

    PAGE_TSX_REFACTOR_PROMPT = """
You are now in the page refactoring phase. Your task is to generate the complete code for the dynamic page file located at `src/app/**/[[...slug]]/page.tsx`.

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
    - Map over `page?.page_components?.map((block, index) => renderComponent(block, index))`.

Now, using the original file content and the mapping as context, generate the complete, refactore
"""
    try:
        schema_content = schema_path.read_text(encoding='utf-8')
        mapping_file_path = project_path.parent / "pages-components-list.txt"
        component_mapping_content = mapping_file_path.read_text(encoding='utf-8')

        initial_input = f"""
        Begin the Contentstack integration process for the Next.js project located at `{project_path}`.

        **CRITICAL INSTRUCTION 1: Core File Snippets**
        When you reach Step 2 (Core File Creation), you MUST use the following code snippets exactly as provided for the corresponding files.

        **File: `.env.local`**
        ```
        {ENV_LOCAL_CONTENT}
        ```

        **File: `src/lib/contentstackClient.ts`**
        ```typescript
        {CONTENTSTACK_CLIENT_TS_CODE_SNIPPET}
        ```

        **CRITICAL INSTRUCTION 2: ESLint Configuration**
        In Step 3, you must find the project's ESLint config file (likely `eslint.config.js`) and modify it to add the following rule to disable the 'no-explicit-any' check: `rules: {{ "@typescript-eslint/no-explicit-any": "off" }}`. This is required for the dynamic rendering logic.

        **CRITICAL INSTRUCTION 3: Query Files**
        In Step 5 (Query Generation), you MUST use the following snippets. For `getPage.ts`, you must first determine the page content type UID from the schema and then construct the query using the provided structure.

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
        - Use this structure: `const result = await stack.contentType('YOUR_UID_HERE').entry().query().where('url', QueryOperation.EQUALS, url).find()`
        - Add editable tags using the discovered UID.

        **CRITICAL INSTRUCTION 4: Dynamic Page (`[[...slug]]/page.tsx`) Refactoring**
        In Step 7, when you refactor the dynamic page, you MUST follow these detailed instructions:
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