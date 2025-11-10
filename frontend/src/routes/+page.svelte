<script lang="ts">
  import { afterUpdate } from 'svelte';
  import { messages } from '$lib/stores/messages';
  import { fetchStandardResponse, fetchStreamingResponse } from '$lib/api/chat';
  import ChatInput from '$lib/components/ChatInput.svelte';
  import ChatMessage from '$lib/components/ChatMessage.svelte';

  let isStreaming = true; // Default to streaming
  let chatContainer: HTMLElement;

  async function handleSubmit(event: CustomEvent<string>) {
    const query = event.detail;

    // 1. Add user message
    messages.update((msgs) => [
      ...msgs,
      { id: crypto.randomUUID(), role: 'user', content: query },
    ]);

    // 2. Add blank assistant message
    const assistantMessageId = crypto.randomUUID();
    messages.update((msgs) => [
      ...msgs,
      { id: assistantMessageId, role: 'assistant', content: '', streaming: true },
    ]);

    try {
      if (isStreaming) {
        // 3a. Call streaming API
        // This function updates the store directly
        await fetchStreamingResponse(query, assistantMessageId);
      } else {
        // 3b. Call standard API
        const response = await fetchStandardResponse(query);
        messages.update((msgs) =>
          msgs.map((msg) => (msg.id === assistantMessageId ? { ...response, id: assistantMessageId } : msg))
        );
      }
    } catch (e: any) {
      // 4. Handle errors
      messages.update((msgs) =>
        msgs.map((msg) =>
          msg.id === assistantMessageId
            ? { ...msg, content: e.message, streaming: false, error: true }
            : msg
        )
      );
    }
  }

  // Auto-scroll logic
  afterUpdate(() => {
    if (chatContainer) {
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }
  });
</script>

<div class="flex flex-col h-screen bg-gray-200">
  <header class="p-4 bg-white border-b border-gray-300 shadow-sm">
    <div class="max-w-4xl mx-auto flex justify-between items-center">
      <h1 class="text-xl font-bold">Local RAG Chat</h1>
      <label class="flex items-center space-x-2 cursor-pointer">
        <input type="checkbox" bind:checked={isStreaming} class="form-checkbox h-5 w-5" />
        <span class="text-gray-700">Enable Streaming</span>
      </label>
    </div>
  </header>

  <main
    bind:this={chatContainer}
    class="flex-1 overflow-y-auto p-4 space-y-4"
  >
    {#each $messages as message (message.id)}
      <ChatMessage {message} />
    {/each}
  </main>

  <footer class="max-w-4xl mx-auto w-full">
    <ChatInput on:submitMessage={handleSubmit} />
  </footer>
</div>
