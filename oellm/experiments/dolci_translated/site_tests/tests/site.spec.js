// @ts-check
const { test, expect } = require("@playwright/test");

// A-25en card temporarily hidden while the matched-compute re-run is in progress.
const GROUP_LABELS = ["Pre-SFT (base)", "OLMo-3-7B-Instruct-SFT", "A-75en"];
const VISIBLE_GROUPS = ["base", "sft-baseline", "A-75en"];
const CARD_COUNT = VISIBLE_GROUPS.length;

test.describe("oellm-completions site", () => {
    test.beforeEach(async ({ page }) => {
        const errors = [];
        page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
        page.on("console", (msg) => {
            if (msg.type() === "error") errors.push(`console.error: ${msg.text()}`);
        });
        await page.goto("/");
        // wait for completions.json fetch + render
        await expect(page.locator("#meta")).toContainText(/checkpoints/);
        // expose for tests to inspect
        page.errors = errors;
    });

    test("renders header, meta line, and visible model cards", async ({ page }) => {
        await expect(page.locator("h1")).toContainText("Multilingual-Dolci-SFTed");
        await expect(page.locator("h1")).toContainText("Olmo3 Checkpoint Viewer");
        await expect(page.locator("#meta")).toContainText(/\d+ checkpoints/);
        await expect(page.locator("#meta")).toContainText(/prompts/);

        const cards = page.locator(".cards-row .card");
        await expect(cards).toHaveCount(CARD_COUNT);
        for (const label of GROUP_LABELS) {
            await expect(page.locator(".card .card-title", { hasText: label })).toBeVisible();
        }
        // A-25en explicitly hidden for now
        await expect(page.locator(".card.group-A-25en")).toHaveCount(0);
    });

    test("language switcher exposes all 7 EU languages", async ({ page }) => {
        for (const lang of ["French", "German", "Finnish", "Swedish", "Italian", "Spanish", "Czech"]) {
            await expect(page.locator(`#lang-segmented .seg-btn:has-text("${lang}")`)).toBeVisible();
        }
    });

    test("category filter pills toggle and clear", async ({ page }) => {
        // pills present
        const pills = page.locator("#category-pills .pill-cat");
        await expect(pills.first()).toBeVisible();
        const count = await pills.count();
        expect(count).toBeGreaterThanOrEqual(5);

        // pick "Math" - counter should shrink
        const totalBefore = await page.locator("#prompt-counter").textContent();
        await page.locator('#category-pills .pill-cat[data-cat="math"]').click();
        await expect(page.locator('#category-pills .pill-cat[data-cat="math"]')).toHaveClass(/active/);
        await expect(page.locator("#prompt-counter")).toContainText("cat filter");

        // clear restores
        await page.locator("#category-clear").click();
        await expect(page.locator('#category-pills .pill-cat[data-cat="math"]')).not.toHaveClass(/active/);
        await expect(page.locator("#prompt-counter")).not.toContainText("cat filter");
    });

    test("prompt categories render as 'Prompt category: …' below prompt", async ({ page }) => {
        // navigate until we find a prompt with at least one category
        let hasTags = false;
        for (let i = 0; i < 20 && !hasTags; i++) {
            const text = (await page.locator("#prompt-tags").textContent() || "").trim();
            if (text) { hasTags = true; break; }
            await page.locator("#next-prompt").click();
        }
        expect(hasTags).toBe(true);
        await expect(page.locator("#prompt-tags .prompt-tags-label")).toContainText(/Prompt category/i);
        const list = await page.locator("#prompt-tags .prompt-tags-list").textContent();
        expect((list || "").trim().length).toBeGreaterThan(0);
    });

    test("language segmented switcher updates prompt counter", async ({ page }) => {
        // each lang has its own count after stratified-by-category sampling
        await expect(page.locator("#prompt-counter")).toContainText(/^#\d+ of \d+/);
        const before = await page.locator("#prompt-counter").textContent();
        await page.getByRole("button", { name: /German/ }).click();
        const afterDe = await page.locator("#prompt-counter").textContent();
        expect(afterDe).not.toBe(before); // counter resets/changes on lang switch
        await page.getByRole("button", { name: /Finnish/ }).click();
        await expect(page.locator("#prompt-counter")).toContainText(/of \d+/);
    });

    test("prompt nav: prev disabled at start, next moves forward", async ({ page }) => {
        await expect(page.locator("#prev-prompt")).toBeDisabled();
        await expect(page.locator("#prompt-counter")).toContainText("#1 of");
        await page.locator("#next-prompt").click();
        await expect(page.locator("#prompt-counter")).toContainText("#2 of");
        await expect(page.locator("#prev-prompt")).toBeEnabled();
        await page.locator("#prev-prompt").click();
        await expect(page.locator("#prompt-counter")).toContainText("#1 of");
    });

    test("jump-to input updates prompt", async ({ page }) => {
        await page.locator("#prompt-jump").fill("42");
        await page.locator("#prompt-jump").press("Enter");
        await expect(page.locator("#prompt-counter")).toContainText("#42 of");
    });

    test("random button changes the prompt (eventually)", async ({ page }) => {
        const before = await page.locator("#prompt-text").textContent();
        // try up to 6 times to handle the unlikely case random picks the same prompt
        let changed = false;
        for (let i = 0; i < 6 && !changed; i++) {
            await page.locator("#random-prompt").click();
            const after = await page.locator("#prompt-text").textContent();
            if (after && after !== before) changed = true;
        }
        expect(changed).toBe(true);
    });

    test("global step slider updates A-75en step tag but NOT static cards", async ({ page }) => {
        const a75Tag = page.locator(".card.group-A-75en .card-step-tag");
        const baseTag = page.locator(".card.group-base .card-step-tag");
        const sftTag = page.locator(".card.group-sft-baseline .card-step-tag");

        // slider default = 4 (final ckpt) → A-75en step 3998
        await expect(a75Tag).toContainText("step 3998");

        const baseInitial = await baseTag.textContent();
        const sftInitial = await sftTag.textContent();
        expect(sftInitial).toContain("step 3252");

        // move slider to position 0 (early)
        await page.locator("#step-slider").evaluate((el) => {
            el.value = "0";
            el.dispatchEvent(new Event("input", { bubbles: true }));
        });
        await expect(a75Tag).toContainText("step 500");

        // static cards should NOT have changed
        await expect(baseTag).toHaveText(baseInitial);
        await expect(sftTag).toHaveText(sftInitial);

        // counter shouldn't say "tick X/Y" any more, and should mention A-75en
        await expect(page.locator("#step-counter")).not.toContainText("tick");
        await expect(page.locator("#step-counter")).toContainText("A-75en step 500");
        // A-25en should NOT appear in the counter while hidden
        await expect(page.locator("#step-counter")).not.toContainText("A-25en");
    });

    test("model toggles hide and show cards", async ({ page }) => {
        await expect(page.locator(".cards-row .card")).toHaveCount(CARD_COUNT);

        // toggle off Pre-SFT (base)
        await page.locator('#model-toggles .pill[data-group="base"]').click();
        await expect(page.locator(".cards-row .card")).toHaveCount(CARD_COUNT - 1);
        await expect(page.locator(".card.group-base")).toHaveCount(0);

        // toggle off A-75en
        await page.locator('#model-toggles .pill[data-group="A-75en"]').click();
        await expect(page.locator(".cards-row .card")).toHaveCount(CARD_COUNT - 2);
        await expect(page.locator(".card.group-A-75en")).toHaveCount(0);

        // toggle base back on
        await page.locator('#model-toggles .pill[data-group="base"]').click();
        await expect(page.locator(".cards-row .card")).toHaveCount(CARD_COUNT - 1);
        await expect(page.locator(".card.group-base")).toHaveCount(1);
    });

    test("toggling all models off shows empty state", async ({ page }) => {
        for (const g of VISIBLE_GROUPS) {
            await page.locator(`#model-toggles .pill[data-group="${g}"]`).click();
        }
        await expect(page.locator(".cards-row .card")).toHaveCount(0);
        await expect(page.locator(".empty-state")).toBeVisible();
    });

    test("step slider at middle picks correct intermediate A-75en ckpt", async ({ page }) => {
        await page.locator("#step-slider").evaluate((el) => {
            el.value = "2";
            el.dispatchEvent(new Event("input", { bubbles: true }));
        });
        await expect(page.locator(".card.group-A-75en .card-step-tag")).toContainText("step 2500");
    });

    test("static cards have a 'static' visual marker", async ({ page }) => {
        await expect(page.locator(".card.group-base .card-step-tag")).toHaveClass(/static/);
        await expect(page.locator(".card.group-sft-baseline .card-step-tag")).toHaveClass(/static/);
        await expect(page.locator(".card.group-A-75en .card-step-tag")).not.toHaveClass(/static/);
    });

    test("'Completions' section header sits between prompt and cards", async ({ page }) => {
        const head = page.locator(".completions-display-head");
        await expect(head).toBeVisible();
        await expect(head.locator(".completions-display-tag")).toContainText("Completions");
        await expect(head.locator(".completions-display-meta")).toContainText(/\d+ models?/);
        await expect(head.locator(".completions-display-meta")).toContainText(/slider \d+\/\d+/);
    });

    test("each card shows completion text (real or placeholder)", async ({ page }) => {
        const completions = page.locator(".cards-row .completion");
        const count = await completions.count();
        expect(count).toBe(CARD_COUNT);
        for (let i = 0; i < count; i++) {
            const t = await completions.nth(i).textContent();
            // either real generation text OR explicit "not generated yet" placeholder
            expect((t || "").trim().length).toBeGreaterThan(0);
        }
    });

    test("List modal opens, shows full prompts, close button works", async ({ page }) => {
        // hidden initially
        await expect(page.locator("#list-modal")).toBeHidden();
        await expect(page.locator("#list-backdrop")).toBeHidden();

        // open
        await page.getByRole("button", { name: "List" }).click();
        await expect(page.locator("#list-modal")).toBeVisible();
        await expect(page.locator("#list-backdrop")).toBeVisible();

        // contains list items
        const items = page.locator("#prompt-list .browse-item");
        await expect(items.first()).toBeVisible();
        const itemCount = await items.count();
        expect(itemCount).toBeGreaterThan(0);

        // first item text is non-empty (full prompt visible, not just truncated index)
        const firstText = await items.nth(0).locator(".browse-item-text").textContent();
        expect((firstText || "").trim().length).toBeGreaterThan(0);

        // close button works
        await page.locator("#list-close").click();
        await expect(page.locator("#list-modal")).toBeHidden();
        await expect(page.locator("#list-backdrop")).toBeHidden();
    });

    test("List modal closes on backdrop click", async ({ page }) => {
        await page.getByRole("button", { name: "List" }).click();
        await expect(page.locator("#list-modal")).toBeVisible();
        await page.locator("#list-backdrop").click();
        await expect(page.locator("#list-modal")).toBeHidden();
    });

    test("List modal closes on Escape key", async ({ page }) => {
        await page.getByRole("button", { name: "List" }).click();
        await expect(page.locator("#list-modal")).toBeVisible();
        await page.keyboard.press("Escape");
        await expect(page.locator("#list-modal")).toBeHidden();
    });

    test("clicking a list item navigates to that prompt", async ({ page }) => {
        await page.getByRole("button", { name: "List" }).click();
        // click the 7th item
        await page.locator("#prompt-list .browse-item").nth(6).click();
        await expect(page.locator("#list-modal")).toBeHidden();
        await expect(page.locator("#prompt-counter")).toContainText("#7 of");
    });

    test("all visible cards render in a single grid row at viewport ≥1080px", async ({ page }) => {
        const cards = page.locator(".cards-row .card");
        await expect(cards).toHaveCount(CARD_COUNT);
        const tops = await cards.evaluateAll((els) => els.map((e) => e.getBoundingClientRect().top));
        const minTop = Math.min(...tops);
        const maxTop = Math.max(...tops);
        // all cards' top edges should be within a few pixels of each other (same row)
        expect(maxTop - minTop).toBeLessThan(5);
    });

    test("theme toggle switches between light and dark, persists, and is keyboard-shortcut'd", async ({ page }) => {
        // Start: should be one of light or dark; capture initial
        const html = page.locator("html");
        const initial = await html.getAttribute("data-theme");
        expect(["light", "dark"]).toContain(initial);

        // Click toggle → flips
        await page.locator("#theme-toggle").click();
        const after1 = await html.getAttribute("data-theme");
        expect(after1).not.toBe(initial);

        // localStorage persisted
        const stored = await page.evaluate(() => localStorage.getItem("oellm-theme"));
        expect(stored).toBe(after1);

        // 'T' keyboard shortcut also flips
        await page.keyboard.press("t");
        const after2 = await html.getAttribute("data-theme");
        expect(after2).toBe(initial);

        // Reload — persisted theme survives
        await page.reload();
        await expect(page.locator("#meta")).toContainText(/checkpoints/);
        const reloaded = await html.getAttribute("data-theme");
        expect(reloaded).toBe(after2);
    });

    test("hero copy mentions LMArena and 7 EU languages", async ({ page }) => {
        const sub = page.locator(".hero-sub");
        await expect(sub).toContainText(/7 EU languages/i);
        await expect(sub).toContainText("LMArena");
        await expect(sub).toContainText(/cs.*de.*es.*fi.*fr.*it.*sv/);
    });

    test("hero text left-aligns with body sections (controls, prompt-display, main)", async ({ page }) => {
        const heroLeft = await page.locator(".hero h1").evaluate((el) => el.getBoundingClientRect().left);
        const controlsLeft = await page.locator(".controls .control").first().evaluate((el) => el.getBoundingClientRect().left);
        const promptLeft = await page.locator(".prompt-display-head").evaluate((el) => el.getBoundingClientRect().left);
        const cardsLeft = await page.locator(".cards-row").evaluate((el) => el.getBoundingClientRect().left);
        // All body-section left edges should match the hero text's left edge within a couple of px
        expect(Math.abs(controlsLeft - heroLeft)).toBeLessThanOrEqual(2);
        expect(Math.abs(promptLeft - heroLeft)).toBeLessThanOrEqual(2);
        expect(Math.abs(cardsLeft - heroLeft)).toBeLessThanOrEqual(2);
    });

    test("no JS errors during initial render and basic interactions", async ({ page }) => {
        await page.locator("#next-prompt").click();
        await page.locator("#step-slider").evaluate((el) => {
            el.value = "1";
            el.dispatchEvent(new Event("input", { bubbles: true }));
        });
        await page.getByRole("button", { name: "List" }).click();
        await page.locator("#list-close").click();
        expect(page.errors).toEqual([]);
    });
});
