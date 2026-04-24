// @ts-check
const { defineConfig, devices } = require("@playwright/test");

const PORT = parseInt(process.env.SITE_PORT || "8765", 10);
const SITE_DIR = require("path").resolve(__dirname, "../site");

module.exports = defineConfig({
    testDir: "./tests",
    fullyParallel: false,
    forbidOnly: !!process.env.CI,
    retries: 0,
    workers: 1,
    reporter: process.env.CI ? "list" : [["list"], ["html", { open: "never" }]],
    use: {
        baseURL: `http://127.0.0.1:${PORT}`,
        trace: "retain-on-failure",
        screenshot: "only-on-failure",
    },
    projects: [
        {
            name: "chromium",
            use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } },
        },
    ],
    webServer: {
        command: `python3 -m http.server ${PORT} --directory ${SITE_DIR} --bind 127.0.0.1`,
        port: PORT,
        reuseExistingServer: !process.env.CI,
        timeout: 30_000,
    },
});
