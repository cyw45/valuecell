import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

const chartSource = await readFile(
  new URL("./candlestick-chart.tsx", import.meta.url),
  "utf8",
);

test("candlestick chart initializes once and disposes only on unmount", () => {
  assert.match(
    chartSource,
    /chartInstance\.current = echarts\.init\(chartRef\.current\)[\s\S]*?return \(\) => \{[\s\S]*?dispose\(\);[\s\S]*?\}, \[\]\);/,
  );
  assert.equal(chartSource.match(/echarts\.init\(/g)?.length, 1);
  assert.equal(chartSource.match(/\.dispose\(\)/g)?.length, 1);
});

test("candlestick chart updates an existing instance when options change", () => {
  assert.match(
    chartSource,
    /chartInstance\.current\.setOption\(option\);[\s\S]*?\}, \[option\]\);/,
  );
});
