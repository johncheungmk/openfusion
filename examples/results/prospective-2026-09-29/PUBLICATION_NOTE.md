# Artifact serialization correction

The registration commit `ca2c9df64d656616c8bd6bade979e21694248d99` was published before
confirmation generation. Its JSON text was uploaded through a reader that normalized
Windows CRLF line endings to LF. The protocol's SHA-256 values were computed from the
original CRLF files, and those exact original files were uploaded to the experiment host.

During inference, before any hidden grading, a reproduction check detected that the
published JSON bytes did not match the declared hashes. A follow-up commit restores the
original file bytes. JSON data, task identities, prompts, tests, model settings, algorithm,
analysis, and the files used by the running experiment are unchanged. The protocol timestamp
and declared checksums are preserved. Each record continues to identify the original
registration commit.

For reproduction, use the corrected files on this branch or after merge into main. To
independently verify this correction, parse each original and corrected JSON file and compare
the objects, or normalize CRLF to LF and compare the text. The corrected artifact byte hashes
must equal the checksums in the unchanged protocol.

This is a publication serialization correction, not a protocol or task-selection change.
