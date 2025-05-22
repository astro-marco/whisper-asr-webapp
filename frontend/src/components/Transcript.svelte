<script lang="ts">
  import type { WhisperResult, Segment } from "../types/WhisperResult";
  import { getContent } from "../utils/convert";
  import { download } from "../utils/utils";
  import Button from "./Button.svelte";

  export let result: WhisperResult;
  export let inputFileName: string;

  function formatTimestamp(n: number) {
    const totalSeconds = Math.round(n);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes.toString().padStart(2, "0")}:${seconds
      .toString()
      .padStart(2, "0")}`;
  }

  interface ProcessedGroup {
    startTime: number;
    speakerLabel?: string;
    text: string;
  }

  $: processed = (() => {
    if (!result?.segments?.length) return [];
    const groups: ProcessedGroup[] = [];
    let curr: { start: number; speaker?: string; lines: string[] } | null =
      null;

    for (const s of result.segments) {
      const txt = s.text.trim();
      if (!curr || curr.speaker !== s.speaker) {
        if (curr) {
          groups.push({
            startTime: curr.start,
            speakerLabel: curr.speaker?.replace(/^SPEAKER_/, "Speaker "),
            text: curr.lines.join(" "),
          });
        }
        curr = { start: s.start, speaker: s.speaker, lines: [txt] };
      } else {
        curr.lines.push(txt);
      }
    }
    if (curr) {
      groups.push({
        startTime: curr.start,
        speakerLabel: curr.speaker?.replace(/^SPEAKER_/, "Speaker "),
        text: curr.lines.join(" "),
      });
    }
    return groups;
  })();

  let fileType: "txt" | "vtt" | "srt" | "tsv" | "json" = "txt";
</script>

<hr class="my-6" />

<!-- Download controls -->
<div class="flex justify-between items-center mb-6">
  <div class="text-2xl font-bold">Transcription Result</div>
  <div class="flex gap-2 items-center">
    <select class="dark:bg-gray-700 dark:text-white" bind:value={fileType}>
      <option value="txt">Plain text</option>
      <option value="vtt">VTT</option>
      <option value="srt">SRT</option>
      <option value="tsv">TSV</option>
      <option value="json">JSON</option>
    </select>
    <Button
      variant="flat"
      on:click={() =>
        download(`${inputFileName}.${fileType}`, getContent(result, fileType))}
    >
      Download
    </Button>
  </div>
</div>

<div class="space-y-3">
  {#each processed as g}
    <div class="segment-group-grid">
      <span
        class="timestamp-cell font-medium text-gray-500 dark:text-gray-400 select-none"
      >
        {formatTimestamp(g.startTime)}
      </span>
      <div>
        {#if g.speakerLabel}
          <span class="speaker-label font-bold text-sm dark:text-gray-300 mr-1">
            {g.speakerLabel}:
          </span>
        {/if}
        <span>{g.text}</span>
      </div>
    </div>
  {/each}
</div>

<style>
  .segment-group-grid {
    display: grid;
    grid-template-columns: auto 1fr;
    column-gap: 0.5rem;
    row-gap: 0.1rem;
  }
  .timestamp-cell {
    padding-top: 1px;
  }
</style>
