<script lang="ts">
  import type { ChatMessage } from '$lib/types';

  export let message: ChatMessage;

  // Simple markdown-like formatter for code blocks
  function formatContent(content: string) {
    return content
      .replace(/```([\s\S]*?)```/g, '<pre class="bg-gray-800 text-white p-3 rounded-md my-2 overflow-x-auto"><code>$1</code></pre>')
      .replace(/`([^`]+)`/g, '<code class="bg-gray-200 text-red-600 px-1 rounded">$1</code>');
  }
</script>

<div
  class="p-4 rounded-lg max-w-3xl mx-auto"
  class:bg-white={message.role === 'user'}
  class:bg-gray-50={message.role === 'assistant'}
>
  <div class="flex items-start space-x-3">
    <div
      class="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center font-bold text-white"
      class:bg-blue-600={message.role === 'user'}
      class:bg-green-600={message.role === 'assistant'}
    >
      {message.role === 'user' ? 'U' : 'AI'}
    </div>

    <div class="flex-1 min-w-0">
      <div class="prose max-w-none">
        {#if message.error}
          <p class="text-red-500 font-semibold">
            An error occurred: {message.content}
          </p>
        {:else if message.streaming && message.content === ''}
          <div class="animate-pulse">Thinking...</div>
        {:else}
          {@html formatContent(message.content)}
          {#if message.streaming}
            <span class="inline-block w-2 h-4 bg-gray-700 animate-pulse ml-1" />
          {/if}
        {/if}
      </div>

      {#if message.metadata && message.metadata.length > 0}
        <details class="mt-3 text-sm">
          <summary class="cursor-pointer font-medium text-gray-600">
            Retrieved {message.metadata.length} sources
          </summary>
          <ul class="list-disc pl-5 mt-1 text-gray-500">
            {#each message.metadata as source}
              <li class="font-mono truncate">{source.split('/').pop()}</li>
            {/each}
          </ul>
        </details>
      {/if}
    </div>
  </div>
</div>
