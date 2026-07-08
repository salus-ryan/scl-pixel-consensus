// Tests for SCL Pixel Consensus extension
// Run: npx tsx tests/openrouter-scl-merge.test.ts

import {
  parseSCL,
  mergeSCL,
  canonicalLayout,
  textToPixelLayer,
  compositeLayersToGrid,
  renderCompositeGrid,
  findDisagreements,
  SCLPixelComponent,
  type SCLConcept,
} from "../extensions/openrouter-scl-merge.ts";

function stripAnsi(s: string): string {
  return s.replace(/\x1b\[[^m]*m/g, "");
}

// ── Test 1: parseSCL ─────────────────────────────────────────────────────────

function testParseSCL() {
  const input = `@president [
  country → United_States
  person → Lincoln
  uncertainty → 0.95
]`;
  const concepts = parseSCL(input);
  if (concepts.length !== 1) throw new Error(`Expected 1 concept, got ${concepts.length}`);
  if (concepts[0].name !== "president") throw new Error(`Expected "president"`);
  if (concepts[0].props.get("person") !== "Lincoln") throw new Error("Expected Lincoln");
  if (concepts[0].confidenceSum !== 0.95) throw new Error(`Expected 0.95`);
  console.log("✓ testParseSCL");
}

// ── Test 2: canonical layout determinism ─────────────────────────────────────

function testCanonicalLayoutDeterminism() {
  // Two different orderings of the same concepts → same canonical layout
  const a: SCLConcept[] = [
    { name: "weather", props: new Map([["temp", "hot"], ["sky", "clear"], ["uncertainty", "0.8"]]), confidenceSum: 0.8, count: 1 },
    { name: "location", props: new Map([["city", "Austin"], ["uncertainty", "0.9"]]), confidenceSum: 0.9, count: 1 },
  ];
  const b: SCLConcept[] = [
    { name: "location", props: new Map([["uncertainty", "0.9"], ["city", "Austin"]]), confidenceSum: 0.9, count: 1 },
    { name: "weather", props: new Map([["sky", "clear"], ["uncertainty", "0.8"], ["temp", "hot"]]), confidenceSum: 0.8, count: 1 },
  ];

  const layoutA = canonicalLayout(a);
  const layoutB = canonicalLayout(b);

  if (layoutA.length !== layoutB.length)
    throw new Error(`Layouts differ in line count: ${layoutA.length} vs ${layoutB.length}`);
  for (let i = 0; i < layoutA.length; i++) {
    if (layoutA[i] !== layoutB[i])
      throw new Error(`Line ${i} differs:\n  ${layoutA[i]}\n  ${layoutB[i]}`);
  }

  // Props should be sorted alphabetically, uncertainty last within a concept
  // Check within @weather block specifically
  const weatherBlock = layoutA.slice(layoutA.indexOf("@weather ["));
  const weatherLines = weatherBlock.slice(0, weatherBlock.indexOf("]") + 1);
  const wText = weatherLines.join("\n");
  const skyIdx = wText.indexOf("sky →");
  const tempIdx = wText.indexOf("temp →");
  const uncIdx = wText.indexOf("uncertainty →");
  if (skyIdx > tempIdx) throw new Error("Props not sorted: sky should come before temp");
  if (tempIdx > uncIdx) throw new Error("Uncertainty should be last within @weather");

  console.log("✓ testCanonicalLayoutDeterminism");
}

// ── Test 3: pixel layer creation ─────────────────────────────────────────────

function testPixelLayer() {
  const lines = ["@test [", "  k → v", "]"];
  const layer = textToPixelLayer(lines);

  // Row 0: @test [
  if (!layer.has(0)) throw new Error("Row 0 missing");
  if (layer.get(0)!.get(0) !== "@") throw new Error("Expected @ at 0,0");
  if (layer.get(0)!.get(1) !== "t") throw new Error("Expected t at 0,1");

  // Spaces should NOT be pixels
  if (layer.get(1)?.has(0)) throw new Error("Space at 1,0 should not be a pixel");
  if (layer.get(1)?.has(1)) throw new Error("Space at 1,1 should not be a pixel");
  // But k at col 2 should be
  if (layer.get(1)?.get(2) !== "k") throw new Error("Expected k at 1,2");

  console.log("✓ testPixelLayer");
}

// ── Test 4: unanimous agreement → white pixels ───────────────────────────────

function testUnanimousAgreement() {
  const concepts: SCLConcept[] = [
    { name: "president", props: new Map([["country", "US"], ["uncertainty", "0.9"]]), confidenceSum: 0.9, count: 1 },
  ];
  const layout = canonicalLayout(concepts);
  const layerA = textToPixelLayer(layout);
  const layerB = textToPixelLayer(layout);
  const layerC = textToPixelLayer(layout);

  const grid = compositeLayersToGrid(
    [
      { layer: layerA, colorIdx: 0 },
      { layer: layerB, colorIdx: 1 },
      { layer: layerC, colorIdx: 2 },
    ],
    3,
  );

  // Every pixel should be white (255,255,255) because all 3 models agree
  for (const [_key, px] of grid.pixels) {
    if (px.rgb.r !== 255 || px.rgb.g !== 255 || px.rgb.b !== 255) {
      throw new Error(
        `Expected white (unanimous) at pixel '${px.char}', got rgb(${px.rgb.r},${px.rgb.g},${px.rgb.b}) with ratio ${px.agreementRatio}`,
      );
    }
    if (px.agreementRatio !== 1.0) {
      throw new Error(`Expected agreement ratio 1.0, got ${px.agreementRatio}`);
    }
  }

  console.log("✓ testUnanimousAgreement");
}

// ── Test 5: single model → raw model color ───────────────────────────────────

function testSingleModelColor() {
  const concepts: SCLConcept[] = [
    { name: "solo", props: new Map([["k", "v"], ["uncertainty", "0.5"]]), confidenceSum: 0.5, count: 1 },
  ];
  const layout = canonicalLayout(concepts);
  const layer = textToPixelLayer(layout);

  const grid = compositeLayersToGrid(
    [{ layer, colorIdx: 0 }],
    3, // 3 total models, but only 1 contributed
  );

  // Pixels should be model 0's color (red: 231,76,60), not white
  for (const [_key, px] of grid.pixels) {
    if (px.rgb.r === 255 && px.rgb.g === 255 && px.rgb.b === 255) {
      throw new Error("Single model should NOT be white (unanimous requires all models)");
    }
    if (px.agreementRatio > 0.34) {
      throw new Error(`Expected ~0.33 ratio for 1/3, got ${px.agreementRatio}`);
    }
  }

  console.log("✓ testSingleModelColor");
}

// ── Test 6: disagreement detection ───────────────────────────────────────────

function testDisagreement() {
  // Model A: @president [ person → Lincoln ]
  // Model B: @president [ person → Biden ]
  // The "person → " part is identical, but the value differs
  const conceptA: SCLConcept[] = [
    { name: "president", props: new Map([["person", "Lincoln"], ["uncertainty", "0.9"]]), confidenceSum: 0.9, count: 1 },
  ];
  const conceptB: SCLConcept[] = [
    { name: "president", props: new Map([["person", "Biden"], ["uncertainty", "0.9"]]), confidenceSum: 0.9, count: 1 },
  ];

  const layerA = textToPixelLayer(canonicalLayout(conceptA));
  const layerB = textToPixelLayer(canonicalLayout(conceptB));

  const layers = [
    { layer: layerA, colorIdx: 0 },
    { layer: layerB, colorIdx: 1 },
  ];

  const disagreements = findDisagreements(layers);

  // "Lincoln" vs "Biden" differ starting at some column
  if (disagreements.length === 0) {
    throw new Error("Expected disagreements where Lincoln vs Biden differ");
  }

  // The structural parts (@president, [, person, →) should be identical
  // so disagreements should only be in the value columns
  const layoutA = canonicalLayout(conceptA);
  const layoutB = canonicalLayout(conceptB);
  const personLineA = layoutA.find((l) => l.includes("person"))!;
  const personLineB = layoutB.find((l) => l.includes("person"))!;

  // "person → " prefix is the same, difference starts at the value
  const prefixLen = "  person → ".length;
  if (personLineA.slice(0, prefixLen) !== personLineB.slice(0, prefixLen)) {
    throw new Error("Structural prefix should be identical before the value");
  }

  console.log("✓ testDisagreement");
}

// ── Test 7: agreement darkens toward white ───────────────────────────────────

function testPartialAgreement() {
  const concepts: SCLConcept[] = [
    { name: "fact", props: new Map([["k", "v"], ["uncertainty", "0.8"]]), confidenceSum: 0.8, count: 1 },
  ];
  const layout = canonicalLayout(concepts);

  // 2 out of 4 models agree
  const grid = compositeLayersToGrid(
    [
      { layer: textToPixelLayer(layout), colorIdx: 0 },
      { layer: textToPixelLayer(layout), colorIdx: 1 },
    ],
    4,
  );

  for (const [_key, px] of grid.pixels) {
    // 2/4 = 50% — should NOT be full white (that's unanimous)
    if (px.rgb.r === 255 && px.rgb.g === 255 && px.rgb.b === 255) {
      throw new Error("2/4 agreement should not be white");
    }
    if (px.agreementRatio !== 0.5) {
      throw new Error(`Expected 0.5 ratio, got ${px.agreementRatio}`);
    }
  }

  console.log("✓ testPartialAgreement");
}

// ── Test 8: renderCompositeGrid produces ANSI output ─────────────────────────

function testRenderOutput() {
  const concepts: SCLConcept[] = [
    { name: "test", props: new Map([["k", "v"], ["uncertainty", "0.5"]]), confidenceSum: 0.5, count: 1 },
  ];
  const layout = canonicalLayout(concepts);
  const grid = compositeLayersToGrid(
    [{ layer: textToPixelLayer(layout), colorIdx: 0 }],
    1,
  );
  const rendered = renderCompositeGrid(grid, 80);

  if (rendered.length === 0) throw new Error("Expected rendered lines");

  const raw = rendered.join("\n");
  if (!raw.includes("\x1b[38;2;")) throw new Error("Expected true-color ANSI codes");

  const plain = stripAnsi(raw);
  if (!plain.includes("@test")) throw new Error("Expected @test in output");

  console.log("✓ testRenderOutput");
}

// ── Test 9: full component render ────────────────────────────────────────────

function testComponentRender() {
  const streams = [
    {
      modelId: "a/model-red:free",
      buffer: "@topic [\n  domain → physics\n  uncertainty → 0.9\n]",
      concepts: parseSCL("@topic [\n  domain → physics\n  uncertainty → 0.9\n]"),
      done: true, tokenCount: 15, colorIdx: 0,
    },
    {
      modelId: "b/model-blue:free",
      buffer: "@topic [\n  domain → physics\n  uncertainty → 0.8\n]",
      concepts: parseSCL("@topic [\n  domain → physics\n  uncertainty → 0.8\n]"),
      done: true, tokenCount: 12, colorIdx: 1,
    },
  ];

  const mockTui = { requestRender: () => {} };
  const comp = new SCLPixelComponent(mockTui as any, {}, () => {}, streams, "test");
  const rendered = comp.render(92);
  const raw = rendered.join("\n");
  const plain = stripAnsi(raw);

  if (!plain.includes("SCL Pixel Consensus")) throw new Error("Missing header");
  if (!plain.includes("composited")) throw new Error("Missing composited section");
  if (!plain.includes("unanimous")) throw new Error("Missing agreement stats");
  if (!plain.includes("model-red")) throw new Error("Missing model-red in legend");
  if (!plain.includes("model-blue")) throw new Error("Missing model-blue in legend");

  console.log("✓ testComponentRender");
}

// ── Test 10: identical SCL collapses to same pixels ──────────────────────────

function testIdenticalSCLCollapse() {
  // The core property: same canonical SCL → same pixels → physical overlap
  const scl = `@knowledge [
  domain → artificial_intelligence
  scope → broad
  uncertainty → 0.85
]`;
  const concepts = parseSCL(scl);

  // 5 models produce identical canonical SCL
  const layers = [];
  for (let i = 0; i < 5; i++) {
    layers.push({
      layer: textToPixelLayer(canonicalLayout(concepts)),
      colorIdx: i,
    });
  }

  const grid = compositeLayersToGrid(layers, 5);
  const disagreements = findDisagreements(layers);

  // Zero disagreements — all models align perfectly
  if (disagreements.length !== 0) {
    throw new Error(`Expected 0 disagreements, got ${disagreements.length}`);
  }

  // Every pixel is unanimous white
  for (const [_key, px] of grid.pixels) {
    if (px.rgb.r !== 255 || px.rgb.g !== 255 || px.rgb.b !== 255) {
      throw new Error(`Expected white at '${px.char}', got rgb(${px.rgb.r},${px.rgb.g},${px.rgb.b})`);
    }
  }

  console.log("✓ testIdenticalSCLCollapse — same meaning → same pixels → white consensus");
}

// ── Test 11: backward compat mergeSCL ────────────────────────────────────────

function testMergeSCLCompat() {
  const streams = [
    { modelId: "a", buffer: "", concepts: [
      { name: "c", props: new Map([["key", "val"]]), confidenceSum: 0.8, count: 1 },
    ], done: true, tokenCount: 5, colorIdx: 0 },
  ];
  const { concepts, contributing } = mergeSCL(streams);
  if (contributing !== 1) throw new Error(`Expected 1, got ${contributing}`);
  if (!concepts.has("c")) throw new Error("Expected concept c");
  console.log("✓ testMergeSCLCompat");
}

// ── Test 12: empty input ─────────────────────────────────────────────────────

function testEmpty() {
  if (parseSCL("").length !== 0) throw new Error("Expected 0");
  if (canonicalLayout([]).length !== 0) throw new Error("Expected 0 layout lines");
  console.log("✓ testEmpty");
}

// ── Run ───────────────────────────────────────────────────────────────────────

function run() {
  console.log("SCL Pixel Consensus tests\n");
  try {
    testParseSCL();
    testCanonicalLayoutDeterminism();
    testPixelLayer();
    testUnanimousAgreement();
    testSingleModelColor();
    testDisagreement();
    testPartialAgreement();
    testRenderOutput();
    testComponentRender();
    testIdenticalSCLCollapse();
    testMergeSCLCompat();
    testEmpty();
    console.log("\n✅ All 12 tests passed!");
    process.exit(0);
  } catch (e) {
    console.error(`\n❌ ${e instanceof Error ? e.message : String(e)}`);
    process.exit(1);
  }
}

run();
