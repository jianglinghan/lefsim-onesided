<!--
  The stage: loops as arcs over the lattice, the two CTCF lanes, and any
  markers the visible view puts on the lattice (e.g. probes).

  CTCFs are edited here: click an empty spot in a lane to add one facing that
  way, drag one to move it, or drag it out of the lanes to remove it. Clicking
  a CTCF opens a popover to set its strength, flip it or remove it. Each
  chevron's opacity shows its strength. The bottom lanes hold the Pol II genes
  and the LEF loading sites the same way: click to add one, drag to move,
  click for a popover.
-->
<script lang="ts">
  import { onMount } from "svelte";

  import { colors, drawTadBands, fit, fmt, MARGIN, onThemeChange, placeTooltip, xScale } from "../render.ts";
  import type { Run } from "./run.svelte.ts";

  let { run }: { run: Run } = $props();

  /**
   * Layout, in CSS pixels: the arcs down to the lattice line at ARCS, a thin
   * strip for the view's markers (probes) just under it, then the two CTCF
   * lanes. The strip is always there, so the stage keeps one height.
   */
  const ARCS = 190;
  const MARKERS = 16;
  const MARKER_Y = ARCS + MARKERS / 2;
  const LANE = 22;
  const LANES_TOP = ARCS + MARKERS;
  /** The Pol II gene lane sits under the two CTCF lanes. */
  const GENES_TOP = LANES_TOP + 2 * LANE;
  /** The LEF loading-site lane sits at the bottom. */
  const LOADING_TOP = GENES_TOP + LANE;

  let canvas: HTMLCanvasElement;
  let tooltip: HTMLDivElement;
  let ctx: CanvasRenderingContext2D | undefined;
  let width = $state(0);
  /**
   * The CTCF under a press (index into run.ctcfs): whether it has moved far
   * enough to be a drag rather than a click, and whether it's out of the lanes
   * (to be removed). Or a marker being dragged.
   */
  let drag = $state<
    | { ctcf: number; x0: number; y0: number; moved: boolean; out: boolean }
    | { gene: number; x0: number; y0: number; moved: boolean; out: boolean }
    | { loading: number; x0: number; y0: number; moved: boolean; out: boolean }
    | { marker: number }
    | undefined
  >(undefined);
  /** The CTCF whose popover is open. */
  let selected = $state<number | undefined>(undefined);
  /** The gene whose popover is open (index into run.genes). */
  let selectedGene = $state<number | undefined>(undefined);
  /** The loading site whose popover is open (index into run.loadingSites). */
  let selectedLoading = $state<number | undefined>(undefined);
  /** A press that moves less than this (CSS px) is a click. */
  const CLICK_SLOP = 4;
  let hover = $state<{ ctcf: number } | { gene: number } | { loading: number } | { marker: number } | undefined>(
    undefined,
  );
  let themeRevision = $state(0);

  const height = LOADING_TOP + LANE + 4;

  /** The lane a CTCF facing this way sits in: capturing from the left on top. */
  const laneTop = (left: boolean) => LANES_TOP + (left ? 0 : LANE);

  function siteAtX(mx: number): number {
    const inner = width - MARGIN.left - MARGIN.right;
    const s = Math.round(((mx - MARGIN.left) / inner) * (run.nSites - 1));
    return Math.max(1, Math.min(run.nSites - 2, s));
  }

  function draw() {
    if (!ctx) return;
    const x = xScale(width);
    const f = run.frame;
    ctx.clearRect(0, 0, width, height);
    drawTadBands(ctx, x, MARGIN.top, ARCS);

    // The lattice.
    ctx.strokeStyle = colors.ink2;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x(0), ARCS + 0.5);
    ctx.lineTo(x(run.nSites - 1), ARCS + 0.5);
    ctx.stroke();

    // The lanes: the chevrons in each say which way it faces.
    for (const left of [true, false]) {
      ctx.fillStyle = colors.band;
      ctx.fillRect(x(0), laneTop(left) + 2, x(run.nSites - 1) - x(0), LANE - 4);
    }
    run.ctcfs.forEach((c, i) => {
      const top = laneTop(c.left);
      const active =
        selected === i ||
        (drag && "ctcf" in drag && drag.ctcf === i) ||
        (hover && "ctcf" in hover && hover.ctcf === i);
      const size = active ? 16 : 12;
      const dir = c.left ? -1 : 1;
      const px = x(c.site);
      const cy = top + LANE / 2;
      // A faint stem up to the lattice, so the site reads against the arcs.
      ctx!.fillStyle = colors.rule;
      ctx!.fillRect(Math.round(px), ARCS + 1, 1, top - ARCS + 2);
      // Opacity shows strength; a CTCF dragged out of the lanes fades further.
      const removing = drag && "ctcf" in drag && drag.ctcf === i && drag.out;
      ctx!.globalAlpha = (0.2 + 0.8 * c.strength) * (removing ? 0.3 : 1);
      ctx!.fillStyle = colors.ink;
      ctx!.beginPath();
      ctx!.moveTo(px + dir * size * 0.6, cy);
      ctx!.lineTo(px, cy - size / 2);
      ctx!.lineTo(px, cy + size / 2);
      ctx!.closePath();
      ctx!.fill();
      ctx!.globalAlpha = 1;
    });

    // The Pol II gene lane: one triangle per TSS, filled while an RNAP is
    // loaded on it.
    ctx.fillStyle = colors.band;
    ctx.fillRect(x(0), GENES_TOP + 2, x(run.nSites - 1) - x(0), LANE - 4);
    run.genes.forEach((g, i) => {
      const active =
        selectedGene === i ||
        (drag && "gene" in drag && drag.gene === i) ||
        (hover && "gene" in hover && hover.gene === i);
      const size = active ? 15 : 11;
      const px = x(g);
      const cy = GENES_TOP + LANE / 2;
      // A faint stem up to the lattice, like the CTCFs have.
      ctx!.fillStyle = colors.rule;
      ctx!.fillRect(Math.round(px), ARCS + 1, 1, GENES_TOP - ARCS + 2);
      const removing = drag && "gene" in drag && drag.gene === i && drag.out;
      ctx!.globalAlpha = removing ? 0.3 : 1;
      ctx!.beginPath();
      ctx!.moveTo(px, cy - size / 2);
      ctx!.lineTo(px - size / 2, cy + size / 2);
      ctx!.lineTo(px + size / 2, cy + size / 2);
      ctx!.closePath();
      if (f?.rnaps.includes(g)) {
        ctx!.fillStyle = colors.rnap;
        ctx!.fill();
      } else {
        ctx!.strokeStyle = colors.rnap;
        ctx!.lineWidth = 1.5;
        ctx!.stroke();
        ctx!.lineWidth = 1;
      }
      ctx!.globalAlpha = 1;
    });

    // The LEF loading lane: a red bar per site, like the figure's track.
    ctx.fillStyle = colors.band;
    ctx.fillRect(x(0), LOADING_TOP + 2, x(run.nSites - 1) - x(0), LANE - 4);
    run.loadingSites.forEach((s, i) => {
      const active =
        selectedLoading === i ||
        (drag && "loading" in drag && drag.loading === i) ||
        (hover && "loading" in hover && hover.loading === i);
      const px = x(s);
      const cy = LOADING_TOP + LANE / 2;
      ctx!.fillStyle = colors.rule;
      ctx!.fillRect(Math.round(px), ARCS + 1, 1, LOADING_TOP - ARCS + 2);
      const removing = drag && "loading" in drag && drag.loading === i && drag.out;
      ctx!.globalAlpha = removing ? 0.3 : 1;
      ctx!.fillStyle = colors.loading;
      const half = active ? 8 : 6;
      ctx!.fillRect(px - 1.25, cy - half, 2.5, 2 * half);
      ctx!.globalAlpha = 1;
    });

    // The view's markers.
    const markers = run.markers;
    if (markers) {
      markers.sites.forEach((s, i) => {
        const px = x(s);
        const active =
          (drag && "marker" in drag && drag.marker === i) || (hover && "marker" in hover && hover.marker === i);
        ctx!.strokeStyle = colors.probe;
        ctx!.setLineDash([2, 3]);
        ctx!.beginPath();
        ctx!.moveTo(px, MARGIN.top);
        ctx!.lineTo(px, MARKER_Y);
        ctx!.stroke();
        ctx!.setLineDash([]);
        ctx!.fillStyle = colors.probe;
        ctx!.beginPath();
        ctx!.arc(px, MARKER_Y, active ? 6 : 4.5, 0, 2 * Math.PI);
        ctx!.fill();
      });
    }

    if (!f) return;
    // Loops as half-ellipses whose height grows with span; held loops on top.
    const maxH = ARCS - MARGIN.top - 4;
    const full = 0.3 * run.nSites;
    ctx.lineWidth = 2;
    for (const pass of [0, 1]) {
      ctx.strokeStyle = pass ? colors.held : colors.extruding;
      ctx.beginPath();
      for (let i = 0; i < f.leftPos.length; i++) {
        if (f.held[i] !== pass) continue;
        const x1 = x(f.leftPos[i]!);
        const x2 = x(f.rightPos[i]!);
        const r = (x2 - x1) / 2;
        const h = Math.min(maxH, Math.max(2, (maxH * (f.rightPos[i]! - f.leftPos[i]!)) / full));
        ctx.moveTo(x1, ARCS);
        ctx.ellipse(x1 + r, ARCS, Math.max(r, 0.5), h, 0, Math.PI, 2 * Math.PI);
      }
      ctx.stroke();
    }
    for (let i = 0; i < f.leftPos.length; i++) {
      ctx.fillStyle = f.held[i] ? colors.held : colors.extruding;
      ctx.fillRect(x(f.leftPos[i]!) - 1, ARCS - 4, 2, 5);
      ctx.fillRect(x(f.rightPos[i]!) - 1, ARCS - 4, 2, 5);
    }
  }

  // Redraw on any change the drawing reads.
  $effect(() => {
    void [
      run.frame,
      run.revision,
      run.markers,
      run.ctcfs.length,
      run.genes.length,
      run.loadingSites.length,
      drag,
      hover,
      selected,
      selectedGene,
      selectedLoading,
      width,
      height,
      themeRevision,
    ];
    if (ctx && canvas.height !== Math.round(height * (devicePixelRatio || 1))) ctx = fit(canvas);
    draw();
  });

  // ---------------------------------------------------------------- pointer

  function point(e: PointerEvent) {
    const rect = canvas.getBoundingClientRect();
    return { mx: e.clientX - rect.left, my: e.clientY - rect.top };
  }

  /** What's under the pointer: a CTCF (index), a gene (index), a loading site (index) or a marker (index), or which lane is empty there. */
  function hit(
    mx: number,
    my: number,
  ):
    | { ctcf: number }
    | { gene: number }
    | { loading: number }
    | { marker: number }
    | { lane: boolean }
    | { geneLane: true }
    | { loadingLane: true }
    | undefined {
    const x = xScale(width);
    if (run.markers && my > ARCS - 4 && my < LANES_TOP) {
      let best = -1;
      let bestPx = 14;
      run.markers.sites.forEach((s, i) => {
        const d = Math.abs(x(s) - mx);
        if (d < bestPx) [best, bestPx] = [i, d];
      });
      return best >= 0 ? { marker: best } : undefined;
    }
    if (my >= LOADING_TOP && my < LOADING_TOP + LANE) {
      let best = -1;
      let bestPx = 10;
      run.loadingSites.forEach((s, i) => {
        const d = Math.abs(x(s) - mx);
        if (d < bestPx) [best, bestPx] = [i, d];
      });
      return best >= 0 ? { loading: best } : { loadingLane: true };
    }
    if (my >= GENES_TOP && my < GENES_TOP + LANE) {
      let best = -1;
      let bestPx = 10;
      run.genes.forEach((g, i) => {
        const d = Math.abs(x(g) - mx);
        if (d < bestPx) [best, bestPx] = [i, d];
      });
      return best >= 0 ? { gene: best } : { geneLane: true };
    }
    if (my < LANES_TOP || my >= LANES_TOP + 2 * LANE) return undefined;
    const left = my < LANES_TOP + LANE;
    let best = -1;
    let bestPx = 10;
    run.ctcfs.forEach((c, i) => {
      if (c.left !== left) return;
      const d = Math.abs(x(c.site) - mx);
      if (d < bestPx) [best, bestPx] = [i, d];
    });
    return best >= 0 ? { ctcf: best } : { lane: left };
  }

  function pointerdown(e: PointerEvent) {
    const { mx, my } = point(e);
    const h = hit(mx, my);
    if (!h) {
      selected = undefined;
      selectedGene = undefined;
      selectedLoading = undefined;
      return;
    }
    if ("loadingLane" in h) {
      // Add a loading site here and keep dragging it.
      selected = undefined;
      selectedGene = undefined;
      selectedLoading = undefined;
      if (!run.addLoadingSite(siteAtX(mx))) return;
      drag = { loading: run.loadingSites.length - 1, x0: mx, y0: my, moved: true, out: false };
    } else if ("loading" in h) {
      selected = undefined;
      selectedGene = undefined;
      drag = { loading: h.loading, x0: mx, y0: my, moved: false, out: false };
    } else if ("geneLane" in h) {
      // Add a gene here and keep dragging it.
      selected = undefined;
      selectedGene = undefined;
      selectedLoading = undefined;
      if (!run.addGene(siteAtX(mx))) return;
      drag = { gene: run.genes.length - 1, x0: mx, y0: my, moved: true, out: false };
    } else if ("gene" in h) {
      selected = undefined;
      selectedLoading = undefined;
      drag = { gene: h.gene, x0: mx, y0: my, moved: false, out: false };
    } else if ("lane" in h) {
      // Add a CTCF here, facing the lane's way, and keep dragging it.
      selected = undefined;
      selectedGene = undefined;
      selectedLoading = undefined;
      if (!run.addCtcf(siteAtX(mx), h.lane)) return;
      drag = { ctcf: run.ctcfs.length - 1, x0: mx, y0: my, moved: true, out: false };
    } else if ("ctcf" in h) {
      selectedGene = undefined;
      selectedLoading = undefined;
      drag = { ctcf: h.ctcf, x0: mx, y0: my, moved: false, out: false };
    } else {
      selected = undefined;
      selectedGene = undefined;
      selectedLoading = undefined;
      drag = { marker: h.marker };
    }
    canvas.setPointerCapture(e.pointerId);
    e.preventDefault();
  }

  function pointermove(e: PointerEvent) {
    const { mx, my } = point(e);
    if (drag && "marker" in drag) {
      run.markers?.move(drag.marker, siteAtX(mx));
      return;
    }
    if (drag) {
      if (!drag.moved && Math.hypot(mx - drag.x0, my - drag.y0) < CLICK_SLOP) return;
      if (!drag.moved) {
        drag = { ...drag, moved: true };
        selected = undefined;
        selectedGene = undefined;
        selectedLoading = undefined;
      }
      if ("loading" in drag) {
        const out = my < LOADING_TOP - 24 || my > LOADING_TOP + LANE + 24;
        if (out !== drag.out) drag = { ...drag, out };
        if (!out) run.moveLoadingSite(drag.loading, siteAtX(mx));
        return;
      }
      if ("gene" in drag) {
        const out = my < GENES_TOP - 24 || my > GENES_TOP + LANE + 24;
        if (out !== drag.out) drag = { ...drag, out };
        if (!out) run.moveGene(drag.gene, siteAtX(mx));
        return;
      }
      const out = my < LANES_TOP - 24 || my > LANES_TOP + 2 * LANE + 24;
      if (out !== drag.out) drag = { ...drag, out };
      if (!out) run.moveCtcf(drag.ctcf, siteAtX(mx));
      return;
    }
    const h = hit(mx, my);
    hover = h && !("lane" in h) && !("geneLane" in h) && !("loadingLane" in h) ? h : undefined;
    canvas.style.cursor = hover ? "pointer" : h ? "copy" : "";
    loopTooltip(e, mx, my);
  }

  function pointerup() {
    if (drag && "ctcf" in drag) {
      if (!drag.moved) selected = selected === drag.ctcf ? undefined : drag.ctcf;
      else if (drag.out) run.removeCtcf(drag.ctcf);
    }
    if (drag && "gene" in drag) {
      if (!drag.moved) selectedGene = selectedGene === drag.gene ? undefined : drag.gene;
      else if (drag.out) run.removeGene(drag.gene);
    }
    if (drag && "loading" in drag) {
      if (!drag.moved) selectedLoading = selectedLoading === drag.loading ? undefined : drag.loading;
      else if (drag.out) run.removeLoadingSite(drag.loading);
    }
    drag = undefined;
  }

  // The popover's CTCF: closed if it's gone (removed, or a new layout).
  const sel = $derived(selected !== undefined ? run.ctcfs[selected] : undefined);
  /** The popover's gene (a site number). */
  const selGene = $derived(selectedGene !== undefined ? run.genes[selectedGene] : undefined);
  /** The popover's loading site (a site number). */
  const selLoading = $derived(selectedLoading !== undefined ? run.loadingSites[selectedLoading] : undefined);
  $effect(() => {
    void run.nSites;
    if (selected !== undefined && selected >= run.ctcfs.length) selected = undefined;
    if (selectedGene !== undefined && selectedGene >= run.genes.length) selectedGene = undefined;
    if (selectedLoading !== undefined && selectedLoading >= run.loadingSites.length) selectedLoading = undefined;
  });

  function onkeydown(e: KeyboardEvent) {
    if (e.key === "Escape") {
      selected = undefined;
      selectedGene = undefined;
      selectedLoading = undefined;
    }
  }

  /** Hovering the arcs: the innermost loop around the pointer. */
  function loopTooltip(e: PointerEvent, mx: number, my: number) {
    const f = run.frame;
    if (!f || my > ARCS) {
      tooltip.hidden = true;
      return;
    }
    const site = siteAtX(mx);
    let best = -1;
    for (let i = 0; i < f.leftPos.length; i++) {
      if (f.leftPos[i]! - 4 <= site && site <= f.rightPos[i]! + 4) {
        if (best < 0 || f.rightPos[i]! - f.leftPos[i]! < f.rightPos[best]! - f.leftPos[best]!) best = i;
      }
    }
    if (best < 0) {
      tooltip.hidden = true;
      return;
    }
    const l = f.leftPos[best]!;
    const r = f.rightPos[best]!;
    placeTooltip(
      tooltip,
      canvas,
      e,
      `LEF ${best}: sites ${fmt.format(l)}–${fmt.format(r)}, ${fmt.format(r - l)} long, ${f.held[best] ? "held at a CTCF" : "extruding"}`,
    );
  }

  onMount(() => {
    const ro = new ResizeObserver(() => {
      width = canvas.getBoundingClientRect().width;
      ctx = fit(canvas);
      draw();
    });
    ro.observe(canvas);
    const unsubscribe = onThemeChange(() => themeRevision++);
    return () => {
      unsubscribe();
      ro.disconnect();
    };
  });
</script>

<div class="stage">
  <canvas
    bind:this={canvas}
    style:height="{height}px"
    onpointerdown={pointerdown}
    onpointermove={pointermove}
    onpointerup={pointerup}
    onpointercancel={pointerup}
    onpointerleave={() => {
      if (!drag) {
        hover = undefined;
        tooltip.hidden = true;
      }
    }}
    aria-label="Loops on the lattice, with CTCF lanes, a Pol II gene lane and a loading-site lane: click a lane to add, drag to move, drag it out of the lane to remove it"
  ></canvas>
  <div class="tooltip" bind:this={tooltip} hidden></div>
  {#if selGene !== undefined && selectedGene !== undefined}
    {@const i = selectedGene}
    <div
      class="popover"
      role="dialog"
      aria-label="Pol II gene at site {fmt.format(selGene)}"
      style:left="{Math.max(8, Math.min(width - 248, xScale(width)(selGene) - 120))}px"
      style:top="{GENES_TOP - 4}px"
    >
      <div class="pop-head">
        <strong>Pol II gene at site {fmt.format(selGene)}</strong>
        <button class="close" onclick={() => (selectedGene = undefined)} aria-label="Close">×</button>
      </div>
      <span class="hint">Loads RNAPs at rate {run.polKin}; drag to move it, drag it out of the lane to remove it.</span>
      <div class="pop-actions">
        <button
          class="btn"
          onclick={() => {
            // `i` reads `selectedGene`, so remove before clearing it.
            run.removeGene(i);
            selectedGene = undefined;
          }}>Remove</button
        >
      </div>
    </div>
  {/if}
  {#if selLoading !== undefined && selectedLoading !== undefined}
    {@const i = selectedLoading}
    <div
      class="popover"
      role="dialog"
      aria-label="LEF loading site at site {fmt.format(selLoading)}"
      style:left="{Math.max(8, Math.min(width - 248, xScale(width)(selLoading) - 120))}px"
      style:top="{LOADING_TOP - 4}px"
    >
      <div class="pop-head">
        <strong>Loading site {fmt.format(selLoading)}</strong>
        <button class="close" onclick={() => (selectedLoading = undefined)} aria-label="Close">×</button>
      </div>
      <span class="hint">A (re)spawning LEF lands here {run.loadingWeight}× as often as on a plain site.</span>
      <div class="pop-actions">
        <button
          class="btn"
          onclick={() => {
            // `i` reads `selectedLoading`, so remove before clearing it.
            run.removeLoadingSite(i);
            selectedLoading = undefined;
          }}>Remove</button
        >
      </div>
    </div>
  {/if}
  {#if sel && selected !== undefined}
    {@const i = selected}
    <div
      class="popover"
      role="dialog"
      aria-label="CTCF at site {fmt.format(sel.site)}"
      style:left="{Math.max(8, Math.min(width - 248, xScale(width)(sel.site) - 120))}px"
      style:top="{laneTop(sel.left) + LANE + 4}px"
    >
      <div class="pop-head">
        <strong>CTCF at site {fmt.format(sel.site)}</strong>
        <button class="close" onclick={() => (selected = undefined)} aria-label="Close">×</button>
      </div>
      <span class="hint">Captures legs arriving from the {sel.left ? "left" : "right"}.</span>
      {#if run.strengthLocked}
        <span class="hint">Strength {sel.strength.toFixed(2)}, shared by all CTCFs. Unlock it on the shelf to set each one.</span>
      {:else}
        <label class="strength">
          <span>Strength</span>
          <output class="mono">{sel.strength.toFixed(2)}</output>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={sel.strength}
            onchange={(e) => run.setCtcfStrength(i, Number(e.currentTarget.value))}
          />
        </label>
      {/if}
      <div class="pop-actions">
        <button class="btn" onclick={() => run.flipCtcf(i)}>Flip direction</button>
        <button
          class="btn"
          onclick={() => {
            // `i` reads `selected`, so remove before clearing it.
            run.removeCtcf(i);
            selected = undefined;
          }}>Remove</button
        >
      </div>
    </div>
  {/if}
</div>

<svelte:window {onkeydown} />

<style>
  .stage {
    position: relative;
  }

  canvas {
    display: block;
    width: 100%;
    touch-action: none;
  }

  .popover {
    position: absolute;
    z-index: 4;
    width: 240px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 10px 12px;
    background: var(--paper);
    border: 1px solid var(--rule);
    border-radius: 10px;
    box-shadow: 0 6px 20px rgb(0 0 0 / 0.14);
    font-size: 14px;
  }

  .pop-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .close {
    border: 0;
    background: none;
    font-size: 18px;
    line-height: 1;
    cursor: pointer;
    color: var(--muted);
  }

  .strength {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 4px 10px;
    color: var(--ink-2);
  }

  .strength input {
    grid-column: 1 / -1;
    width: 100%;
  }

  .pop-actions {
    display: flex;
    gap: 8px;
  }

  .pop-actions .btn {
    font-size: 13px;
    padding: 4px 10px;
  }
</style>
