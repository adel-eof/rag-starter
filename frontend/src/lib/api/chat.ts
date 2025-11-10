import { PUBLIC_API_BASE_URL } from '$env/static/public';
import type { ChatMessage, NonStreamingResponse, StreamingChunk } from '$lib/types';
import { get } from 'svelte/store';
import { messages } from '$lib/stores/messages';

/**
 * Handles the non-streaming GET request
 */
/**
 * Handles the non-streaming POST request
 */
export async function fetchStandardResponse(query: string, top_k = 3): Promise<ChatMessage> {
  // Target the correct non-streaming endpoint
  const url = `${PUBLIC_API_BASE_URL}/v1/query`;

  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    // Send a JSON body as per the working curl test
    body: JSON.stringify({
      query: query,
      top_k: top_k,
    }),
  });

  if (!res.ok) {
    throw new Error(`API request failed with status ${res.status}`);
  }

  const data: NonStreamingResponse = await res.json();

  return {
    id: data.id,
    role: 'assistant',
    content: data.choices[0].message.content,
  };
}

/**
 * Handles the streaming POST request
 */
export async function fetchStreamingResponse(query: string, messageId: string) {
  console.log(`[DEBUG] Starting stream for query: "${query}"`);

  const res = await fetch(`${PUBLIC_API_BASE_URL}/v1/query/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  });

  // --- NEW DEBUG LOGIC ---
  console.log(`[DEBUG] Response Status: ${res.status}`);
  console.log('[DEBUG] Response Headers:');
  res.headers.forEach((value, key) => {
    console.log(`  ${key}: ${value}`);
  });
  // --- END NEW DEBUG LOGIC ---

  if (!res.ok || !res.body) {
    console.error('[DEBUG] API request failed or has no body', res);
    throw new Error(`API request failed with status ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  console.log('[DEBUG] Stream reader created, entering while loop...');

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      console.log('[DEBUG] Stream finished (done=true).');
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    // console.log('[DEBUG] Buffer updated:', JSON.stringify(buffer)); // Uncomment if desperate

    let eolIndex;
    while ((eolIndex = buffer.indexOf('\n\n')) !== -1) {
      const line = buffer.substring(0, eolIndex).trim();
      buffer = buffer.substring(eolIndex + 2);

      // --- THIS IS THE KEY LOG ---
      console.log('[DEBUG] Processing line:', JSON.stringify(line));

      if (line === '') {
        console.log('[DEBUG] Skipping empty line.');
        continue;
      }

      if (line.startsWith('data: [DONE]')) {
        console.log('[DEBUG] Found [DONE] signal.');
        messages.update((msgs) =>
          msgs.map((msg) => (msg.id === messageId ? { ...msg, streaming: false } : msg))
        );
        return;
      }

      if (line.startsWith('data: ')) {
        const jsonString = line.substring('data: '.length);
        console.log('[DEBUG] Parsing JSON:', jsonString);

        if (!jsonString) {
          console.log('[DEBUG] Skipping empty data string.');
          continue;
        }

        try {
          const chunk: StreamingChunk = JSON.parse(jsonString);

          messages.update((msgs) => {
            console.log('[DEBUG] Updating messages store...'); // Check if this fires
            return msgs.map((msg) => {
              if (msg.id !== messageId) return msg;

              if ('metadata' in chunk && chunk.metadata) {
                return { ...msg, metadata: chunk.metadata.retrieved };
              }

              if ('choices' in chunk && chunk.choices[0]?.delta?.content) {
                return { ...msg, content: msg.content + chunk.choices[0].delta.content };
              }

              return msg;
            });
          });
        } catch (e) {
          console.error('[DEBUG] Failed to parse stream chunk:', jsonString, e);
          // ... (error handling) ...
          return;
        }
      } else {
        console.warn('[DEBUG] Line skipped (no "data: " prefix):', JSON.stringify(line));
      }
    }
  }
}
