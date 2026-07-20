import assert from "node:assert/strict";
import test from "node:test";
import { renderToStaticMarkup } from "react-dom/server";
import MarkdownRenderer from "./markdown-renderer.tsx";

test("does not interpret raw HTML from markdown content", () => {
  const html = renderToStaticMarkup(
    <MarkdownRenderer
      content={'<img src="x" onerror="alert(1)"><script>alert(2)</script>'}
    />,
  );

  assert.doesNotMatch(html, /<img|<script/i);
  assert.match(html, /&lt;img/);
});

test("continues to render safe Markdown and GFM", () => {
  const html = renderToStaticMarkup(
    <MarkdownRenderer content={"## Safe\n\n**bold**\n\n- [x] checked"} />,
  );

  assert.match(html, /<h2>Safe<\/h2>/);
  assert.match(html, /<strong>bold<\/strong>/);
  assert.match(html, /type="checkbox" disabled="" checked=""/);
});
