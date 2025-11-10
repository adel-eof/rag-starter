<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  let query = '';
  const dispatch = createEventDispatcher();

  function handleSubmit() {
    if (!query.trim()) return;
    dispatch('submitMessage', query);
    query = '';
  }
</script>

<form
  on:submit|preventDefault={handleSubmit}
  class="flex items-center p-4 bg-gray-100 border-t border-gray-300"
>
  <input
    type="text"
    bind:value={query}
    placeholder="Send a message..."
    class="flex-1 px-4 py-2 border border-gray-400 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
  />
  <button
    type="submit"
    class="ml-3 px-6 py-2 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
  >
    Send
  </button>
</form>
