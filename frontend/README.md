# Svelte Local LLM Chat Frontend

This is a modern, responsive chat frontend built with **SvelteKit**, **Tailwind CSS**, and **TypeScript**. It's designed to connect to a local backend service (like one built with `llama-cpp-python`) and supports both streaming and non-streaming API endpoints.



## ✨ Features

* **Dual API Modes:** Toggle between real-time streaming (SSE) and standard JSON responses.
* **Streaming Support:** Displays partial tokens as they arrive from the `POST /v1/query/stream` endpoint.
* **Metadata Display:** Shows retrieved documents from the RAG pipeline in a collapsible section.
* **Error Handling:** Gracefully displays network or API errors in the chat interface.
* **Responsive Design:** Minimalist, mobile-friendly layout styled with Tailwind CSS.
* **Type-Safe:** Fully written in TypeScript with clear interfaces for API and state.

## 🛠️ Technical Stack

* **Framework:** SvelteKit
* **Styling:** Tailwind CSS
* **Language:** TypeScript
* **API Client:** `fetch` (using `ReadableStream` for streaming)
* **State Management:** Svelte Stores

---

## 🚀 Getting Started

### 1. Prerequisites

* [Node.js](https://nodejs.org/) (v18 or higher)
* A running backend service accessible at `http://localhost:8000`.

### 2. Installation

1.  Clone this repository and navigate into the directory:
    ```bash
    git clone https://your-repo-url/frontend.git
    cd frontend
    ```

2.  Install the dependencies:
    ```bash
    npm install
    ```

### 3. Configuration

The application connects to the backend API using an environment variable.

1.  Create a `.env` file in the root of the project:
    ```bash
    touch .env
    ```

2.  Add your backend's base URL to the `.env` file:
    ```ini
    # .env
    PUBLIC_API_BASE_URL="http://localhost:8000"
    ```

### 4. Running the Development Server

Start the SvelteKit dev server. The app will be available at `http://localhost:5173`.

```bash
npm run dev
```

### 5. Building for Production

To create a production version of the app:

```bash
npm run build
```

You can preview the production build with `npm run preview`.

---

## ⚙️ How It Works

This application is built around a central Svelte store (`src/lib/stores/messages.ts`) that holds an array of `ChatMessage` objects.

### 🔌 API Modes

A toggle in the header controls a boolean state (`isStreaming`). This value determines which API function is called.

#### 1. Streaming Mode (Default)

* **Endpoint:** `POST /v1/query/stream`
* **Handler:** `src/lib/api/chat.ts` -> `fetchStreamingResponse`
* **Method:**
    1.  A `fetch` request is made. The application gets a `ReadableStream` from the response body.
    2.  We manually read from the stream using a `TextDecoder`.
    3.  A custom parser handles the non-standard `data: data: {...}` format provided by the backend.
    4.  As valid JSON chunks (containing metadata or content) are parsed, the corresponding `ChatMessage` object in the Svelte store is updated reactively.
    5.  The stream is closed when a `data: data: [DONE]` signal is received.

#### 2. Non-Streaming Mode

* **Endpoint:** `GET /v1/query/stream?q=...`
* **Handler:** `src/lib/api/chat.ts` -> `fetchStandardResponse`
* **Method:**
    1.  A simple `GET` request is made using `await fetch()`.
    2.  The full JSON response is parsed.
    3.  The blank "assistant" message in the store is replaced with the complete response.
