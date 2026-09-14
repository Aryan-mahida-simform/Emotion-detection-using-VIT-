const IMAGE_EXTENSIONS = /\.(avif|bmp|gif|heic|heif|jpe?g|png|tiff?|webp)$/i;

export function carriesFiles(event) {
  return Array.from(event?.dataTransfer?.types ?? []).includes("Files");
}

export function isImageFile(file) {
  if (file?.type) return file.type.startsWith("image/");
  return IMAGE_EXTENSIONS.test(file?.name ?? "");
}

export function firstFile(event) {
  return event?.dataTransfer?.files?.[0] ?? null;
}
