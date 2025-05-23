<script lang="ts">
  import toast, { Toaster } from "svelte-french-toast";
  import "./app.css";
  import Button from "./components/Button.svelte";
  import DropZone from "./components/DropZone.svelte";
  import LanguageSelect from "./components/LanguageSelect.svelte";
  import ModelSelect from "./components/ModelSelect.svelte";
  import Tooltip from "./components/Tooltip.svelte";
  import Transcript from "./components/Transcript.svelte";
  import CircleQuestion from "./components/icons/CircleQuestion.svelte";
  import type { WhisperResult } from "./types/WhisperResult";
  import { fetchModels } from "./utils/utils";

  let task: string = "transcribe";
  let files: FileList;
  let model: string = "base";
  let result: WhisperResult;
  let language: string = "";
  let initialPrompt: string;
  let wordTimestamps: boolean = false;
  let diarize: boolean = false;
  let numSpeakers: number | null = null;

  let models: Promise<any>;

  let loading = false;

  let transcript: HTMLDivElement;

  let start: number;
  let end: number;

  let lastInputFileName: string;

  let debugJSON: string = "";

  async function submit() {
    if (!files) {
      toast.error("Please upload a file.");
      return;
    }
    const formData = new FormData();
    formData.set("task", task);
    formData.set("file", files.item(0));
    formData.set("model", model);
    formData.set("language", language);
    formData.set("initial_prompt", initialPrompt);
    formData.set("word_timestamps", wordTimestamps.toString());
    formData.set("diarize", diarize.toString());
    if (diarize && numSpeakers && numSpeakers > 0) {
      formData.set("num_speakers", numSpeakers.toString());
    }

    result = undefined;
    loading = true;
    start = Date.now();

    let fileName = files.item(0).name;

    await fetch("/transcribe", {
      method: "POST",
      body: formData,
    })
      .then((res) => res.json())
      .then((json) => {
        result = json;
        debugJSON = JSON.stringify(json, null, 2);
        console.log(
          "Received result from backend:",
          JSON.stringify(result, null, 2),
        ); // Add this line to inspect
        loading = false;
        end = Date.now();
        lastInputFileName = fileName;
        setTimeout(() =>
          transcript?.scrollIntoView({ behavior: "smooth", block: "start" }),
        );
      })
      .catch((err) => {
        loading = false;
        console.error(err);
        toast.error("An error occurred during transcription.");
      });
  }

  $: if (!loading) {
    models = fetchModels();
  }

  $: if (language === "english" || model === "turbo") {
    task = "transcribe";
  }
</script>

<div class="max-w-3xl mx-auto mt-8">
  <main class="space-y-6">
    <div class="space-y-2">
      <h1 class="text-3xl font-bold">Transcription App</h1>
      <p>
        A simple tool for transcribing recordings locally using OpenAI's <a
          href="https://github.com/openai/whisper"
          class="underline font-medium">Whisper</a
        > model.
      </p>
    </div>

    <form
      action="/transcribe"
      method="POST"
      enctype="multipart/form-data"
      on:submit|preventDefault={submit}
      class="space-y-6"
    >
      <DropZone bind:files />

      <ModelSelect bind:model {models} />

      <div class="flex justify-between gap-6">
        <div class="flex flex-col gap-2 w-full">
          <label for="language" class="text-xs uppercase font-bold">
            Select Language
            {#if model.includes(".en")}
              <Tooltip
                tip="You have selected an English-only model. Language selection is disabled."
              >
                <CircleQuestion
                  class="fill-gray-600 dark:fill-gray-400 w-4 h-4 ml-1 -mb-0.5"
                />
              </Tooltip>
            {/if}
          </label>

          <LanguageSelect bind:language english={model.includes(".en")} />
        </div>

        <div class="flex flex-col gap-2 w-full">
          <label for="task" class="text-xs uppercase font-bold">
            Translation
          </label>

          <select
            name="language"
            bind:value={task}
            disabled={language === "english" || model === "turbo"}
            class="dark:bg-gray-700 dark:text-white disabled:text-gray-400"
          >
            <option value="transcribe">Keep as-is</option>
            <option value="translate">Translate to English</option>
          </select>
        </div>

        <div class="flex flex-col gap-2 w-full">
          <p class="text-xs uppercase font-bold mb-0.5">
            Per-word timestamps
            <Tooltip
              tip={"If enabled, the model will attempt to add a timestamp to every individual word. This will increase the required processing time."}
            >
              <CircleQuestion
                class="fill-gray-600 dark:fill-gray-400 w-4 h-4 ml-1 -mb-0.5"
              />
            </Tooltip>
          </p>

          <div class="flex gap-2 h-8 mt-1">
            <div>
              <input
                type="radio"
                name="word_timestamps"
                id="wordTimestampsEnabled"
                value={true}
                bind:group={wordTimestamps}
                class="sr-only peer"
              />
              <label
                for="wordTimestampsEnabled"
                class="peer-checked:bg-gray-200 dark:peer-checked:bg-gray-700 w-full border border-gray-300 dark:border-gray-500 px-3 py-2 rounded-md cursor-pointer transition-colors"
              >
                Enabled
              </label>
            </div>

            <div>
              <input
                type="radio"
                name="word_timestamps"
                id="wordTimestampsDisabled"
                value={false}
                bind:group={wordTimestamps}
                class="sr-only peer"
              />
              <label
                for="wordTimestampsDisabled"
                class="peer-checked:bg-gray-200 dark:peer-checked:bg-gray-700 w-full border border-gray-300 dark:border-gray-500 px-3 py-2 rounded-md cursor-pointer transition-colors"
              >
                Disabled
              </label>
            </div>
          </div>
        </div>

        <div class="flex flex-col gap-2 w-full">
          <label for="diarize" class="text-xs uppercase font-bold">
            Speaker Diarization
          </label>
          <div class="flex items-center gap-2 mt-1">
            <input
              type="checkbox"
              id="diarize"
              bind:checked={diarize}
              class="h-4 w-4"
            />
            <label for="diarize" class="text-sm"
              >Enable speaker diarization</label
            >
          </div>
        </div>

        {#if diarize}
          <div class="flex flex-col gap-2 w-full">
            <label for="numSpeakers" class="text-xs uppercase font-bold">
              Number of Speakers
              <Tooltip
                tip="Optional. Specify the exact number of speakers. Leave blank or set to 0 for auto-detection."
              >
                <CircleQuestion
                  class="fill-gray-600 dark:fill-gray-400 w-4 h-4 ml-1 -mb-0.5"
                />
              </Tooltip>
            </label>
            <input
              type="number"
              id="numSpeakers"
              bind:value={numSpeakers}
              min="0"
              placeholder="Auto-detect"
              class="dark:bg-gray-700 dark:text-white dark:placeholder:text-gray-400 p-2 border rounded-md"
              on:input={() => {
                if (numSpeakers < 0) numSpeakers = 0;
              }}
            />
          </div>
        {/if}
      </div>

      <div class="flex flex-col gap-2 w-full">
        <label for="initial-prompt" class="text-xs uppercase font-bold">
          Initial Prompt
          <Tooltip
            tip={`Optional text to provide as a prompt for the first window. This can be used to provide, or "prompt-engineer" a context for transcription, e.g. custom vocabularies or proper nouns to make it more likely to predict those word correctly.`}
          >
            <CircleQuestion
              class="fill-gray-600 dark:fill-gray-400 w-4 h-4 ml-1 -mb-0.5"
            />
          </Tooltip>
        </label>
        <textarea
          class="col-span-2 dark:text-white dark:placeholder:text-gray-400 dark:bg-gray-700"
          id="initial-prompt"
          bind:value={initialPrompt}
        />
      </div>
      <div class="flex justify-end">
        <Button variant="primary" {loading}>
          {loading ? "Transcribing..." : "Transcribe"}
        </Button>
      </div>
    </form>
    <div bind:this={transcript}>
      {#if result}
        <Transcript {result} inputFileName={lastInputFileName} />
      {/if}
    </div>

    {#if debugJSON}
      <details class="mt-4 p-2 bg-gray-100 dark:bg-gray-800 rounded">
        <summary class="cursor-pointer font-medium">Mostra Debug JSON</summary>
        <pre
          class="whitespace-pre-wrap overflow-auto max-h-64 text-xs">{debugJSON}</pre>
      </details>
    {/if}
  </main>
</div>

<Toaster />
