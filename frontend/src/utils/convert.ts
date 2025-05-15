import type { WhisperResult } from "../types/WhisperResult";

function formatTimestamp(
  seconds: number,
  alwaysIncludeHours = false,
  decimalMarker = ".",
) {
  let milliseconds = Math.round(seconds * 1000.0);

  let hours = Math.floor(milliseconds / 3_600_000);
  milliseconds -= hours * 3_600_000;

  let minutes = Math.floor(milliseconds / 60_000);
  milliseconds -= minutes * 60_000;

  seconds = Math.floor(milliseconds / 1_000);
  milliseconds -= seconds * 1_000;

  let hoursMarker =
    alwaysIncludeHours || hours > 0
      ? hours.toString().padStart(2, "0") + ":"
      : "";

  return `${hoursMarker}${minutes.toString().padStart(2, "0")}:${seconds
    .toString()
    .padStart(2, "0")}${decimalMarker}${milliseconds
    .toString()
    .padStart(3, "0")}`;
}

export function getContent(
  result: WhisperResult,
  format: "txt" | "vtt" | "srt" | "tsv" | "json",
) {
  switch (format) {
    case "txt": {
      // group consecutive segments by speaker
      const groups: { speaker?: string; lines: string[] }[] = [];
      let curr: { speaker?: string; lines: string[] } | null = null;

      for (const seg of result.segments) {
        const txt = seg.text.trim();
        if (!curr || curr.speaker !== seg.speaker) {
          if (curr) groups.push(curr);
          curr = { speaker: seg.speaker, lines: [txt] };
        } else {
          curr.lines.push(txt);
        }
      }
      if (curr) groups.push(curr);

      return groups
        .map(
          (g) =>
            (g.speaker ? g.speaker + ": " : "") + g.lines.join(" ")
        )
        .join("\n");
    }
    case "vtt": {
      return (
        "WEBVTT\n\n" +
        result.segments
          .map((segment) => {
            const speakerPrefix = segment.speaker ? segment.speaker + ": " : "";
            return `${formatTimestamp(segment.start)} --> ${formatTimestamp(
              segment.end,
            )}\n${speakerPrefix}${segment.text.trim().replace("-->", "->")}\n`;
          })
          .join("\n")
      );
    }
    case "srt": {
      return result.segments
        .map((segment, i) => {
          const speakerPrefix = segment.speaker ? segment.speaker + ": " : "";
          return `${i + 1}\n${formatTimestamp(
            segment.start,
            true,
            ",",
          )} --> ${formatTimestamp(segment.end, true, ",")}\n${speakerPrefix}${segment.text
            .trim()
            .replace("-->", "->")}\n`;
        })
        .join("\n");
    }
    case "tsv": {
      return (
        "start\tend\tspeaker\ttext\n" +
        result.segments
          .map((segment) => {
            return `${Math.round(1000 * segment.start)}\t${Math.round(
              1000 * segment.end,
            )}\t${segment.speaker || ""}\t${segment.text.trim().replace("\t", " ")}`;
          })
          .join("\n")
      );
    }
    case "json": {
      return JSON.stringify(result);
    }
  }
}
