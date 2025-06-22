import unittest
import os
import asyncio
import tempfile
from tools.playwright_automation import PlaywrightAutomation

class TestPlaywrightAutomation(unittest.TestCase):
    def setUp(self):
        self.automation = PlaywrightAutomation(headless=True)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
    def tearDown(self):
        self.loop.run_until_complete(self.automation.close())
        self.loop.close()
        
    def test_initialization(self):
        """Test that PlaywrightAutomation initializes correctly."""
        self.assertEqual(self.automation.browser_type, "chromium")
        self.assertTrue(self.automation.headless)
        self.assertEqual(self.automation.slow_mo, 0)
        self.assertTrue(os.path.exists(self.automation.screenshots_dir))
        
    def test_initialize_and_close(self):
        """Test initializing and closing Playwright."""
        result = self.loop.run_until_complete(self.automation.initialize())
        self.assertTrue(result)
        self.assertTrue(self.automation.is_initialized)
        
        self.loop.run_until_complete(self.automation.close())
        self.assertFalse(self.automation.is_initialized)
        
    def test_navigate(self):
        """Test navigating to a URL."""
        self.loop.run_until_complete(self.automation.initialize())
        
        result = self.loop.run_until_complete(
            self.automation.navigate("https://example.com")
        )
        
        self.assertTrue(result["success"])
        self.assertEqual(result["url"], "https://example.com/")
        self.assertEqual(result["status"], 200)
        self.assertTrue(os.path.exists(result["screenshot"]))
        
    def test_get_page_content(self):
        """Test getting page content."""
        self.loop.run_until_complete(self.automation.initialize())
        self.loop.run_until_complete(self.automation.navigate("https://example.com"))
        
        result = self.loop.run_until_complete(self.automation.get_page_content())
        
        self.assertTrue(result["success"])
        self.assertEqual(result["url"], "https://example.com/")
        self.assertIn("Example Domain", result["title"])
        self.assertIn("<html", result["content"])
        self.assertIn("Example Domain", result["text"])
        self.assertTrue(os.path.exists(result["screenshot"]))
        
    def test_click(self):
        """Test clicking an element."""
        self.loop.run_until_complete(self.automation.initialize())
        self.loop.run_until_complete(self.automation.navigate("https://example.com"))
        
        result = self.loop.run_until_complete(
            self.automation.click("a")  # Click the link on example.com
        )
        
        self.assertTrue(result["success"])
        self.assertEqual(result["selector"], "a")
        self.assertTrue(os.path.exists(result["screenshot"]))
        
    def test_type_text(self):
        """Test typing text into an input field."""
        self.loop.run_until_complete(self.automation.initialize())
        self.loop.run_until_complete(
            self.automation.navigate("https://www.google.com")
        )
        
        result = self.loop.run_until_complete(
            self.automation.type_text("input[name='q']", "playwright automation")
        )
        
        self.assertTrue(result["success"])
        self.assertEqual(result["selector"], "input[name='q']")
        self.assertEqual(result["text"], "playwright automation")
        self.assertTrue(os.path.exists(result["screenshot"]))
        
    def test_extract_data(self):
        """Test extracting data from a page."""
        self.loop.run_until_complete(self.automation.initialize())
        self.loop.run_until_complete(self.automation.navigate("https://example.com"))
        
        result = self.loop.run_until_complete(
            self.automation.extract_data({
                "title": "h1",
                "description": "p"
            })
        )
        
        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["title"], "Example Domain")
        self.assertTrue("example" in result["data"]["description"].lower())
        self.assertTrue(os.path.exists(result["screenshot"]))
        
    def test_take_screenshot(self):
        """Test taking a screenshot."""
        self.loop.run_until_complete(self.automation.initialize())
        self.loop.run_until_complete(self.automation.navigate("https://example.com"))
        
        result = self.loop.run_until_complete(
            self.automation.take_screenshot(full_page=True)
        )
        
        self.assertTrue(result["success"])
        self.assertTrue(result["full_page"])
        self.assertTrue(os.path.exists(result["screenshot"]))
        
    def test_execute_javascript(self):
        """Test executing JavaScript on the page."""
        self.loop.run_until_complete(self.automation.initialize())
        self.loop.run_until_complete(self.automation.navigate("https://example.com"))
        
        result = self.loop.run_until_complete(
            self.automation.execute_javascript("return document.title;")
        )
        
        self.assertTrue(result["success"])
        self.assertEqual(result["result"], "Example Domain")
        self.assertTrue(os.path.exists(result["screenshot"]))
        
    def test_fill_form(self):
        """Test filling a form."""
        self.loop.run_until_complete(self.automation.initialize())
        self.loop.run_until_complete(
            self.automation.navigate("https://www.google.com")
        )
        
        result = self.loop.run_until_complete(
            self.automation.fill_form({
                "input[name='q']": "playwright automation"
            })
        )
        
        self.assertTrue(result["success"])
        self.assertEqual(result["form_data"]["input[name='q']"], "playwright automation")
        self.assertTrue(os.path.exists(result["screenshot_before"]))
        
    def test_run_test_script(self):
        """Test running a test script."""
        self.loop.run_until_complete(self.automation.initialize())
        
        test_script = [
            {"type": "navigate", "url": "https://example.com"},
            {"type": "extract", "selectors": {"title": "h1", "description": "p"}},
            {"type": "screenshot", "full_page": True}
        ]
        
        result = self.loop.run_until_complete(
            self.automation.run_test(test_script)
        )
        
        self.assertTrue(result["success"])
        self.assertEqual(len(result["results"]), 3)
        self.assertEqual(result["results"][0]["action_type"], "navigate")
        self.assertEqual(result["results"][1]["action_type"], "extract")
        self.assertEqual(result["results"][2]["action_type"], "screenshot")
        
    def test_wait_for_selector(self):
        """Test waiting for a selector to appear."""
        self.loop.run_until_complete(self.automation.initialize())
        self.loop.run_until_complete(self.automation.navigate("https://example.com"))
        
        result = self.loop.run_until_complete(
            self.automation.wait_for_selector("h1", timeout=5000)
        )
        
        self.assertTrue(result["success"])
        self.assertEqual(result["selector"], "h1")
        self.assertEqual(result["state"], "visible")
        self.assertTrue(os.path.exists(result["screenshot"]))
        
    def test_get_cookies(self):
        """Test getting cookies from the page."""
        self.loop.run_until_complete(self.automation.initialize())
        self.loop.run_until_complete(self.automation.navigate("https://example.com"))
        
        result = self.loop.run_until_complete(
            self.automation.get_cookies()
        )
        
        self.assertTrue(result["success"])
        self.assertIsInstance(result["cookies"], list)
        
    def test_get_network_requests(self):
        """Test getting network requests made by the page."""
        self.loop.run_until_complete(self.automation.initialize())
        self.loop.run_until_complete(self.automation.navigate("https://example.com"))
        
        result = self.loop.run_until_complete(
            self.automation.get_network_requests()
        )
        
        self.assertTrue(result["success"])
        self.assertIsInstance(result["requests"], list)
        
    def test_get_performance_metrics(self):
        """Test getting performance metrics for the page."""
        self.loop.run_until_complete(self.automation.initialize())
        self.loop.run_until_complete(self.automation.navigate("https://example.com"))
        
        result = self.loop.run_until_complete(
            self.automation.get_performance_metrics()
        )
        
        self.assertTrue(result["success"])
        self.assertIn("metrics", result)
        self.assertIn("domInteractive", result["metrics"])

if __name__ == '__main__':
    unittest.main()