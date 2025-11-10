import { writable } from 'svelte/store';
import type { ChatMessage } from '$lib/types';

export const messages = writable<ChatMessage[]>([]);
