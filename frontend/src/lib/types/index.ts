// The internal state for a message in the UI
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  metadata?: string[];
  streaming?: boolean;
  error?: boolean;
}

// API response for GET (non-streaming)
export interface NonStreamingResponse {
  id: string;
  choices: {
    message: {
      role: 'assistant';
      content: string;
    };
  }[];
  // ... other properties
}

// API response for POST (streaming chunk)
// Note: Based on your example, metadata is in its own object
export type StreamingChunk =
  | {
      id: string;
      choices: {
        delta: { content?: string };
        finish_reason: string | null;
      }[];
    }
  | {
      metadata: {
        retrieved: string[];
      };
    };
