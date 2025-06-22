import os
import logging
import json
import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import tempfile
import base64
import re

# Import Playwright
try:
    from playwright.async_api import async_playwright, Page, Browser, BrowserContext, ElementHandle
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PlaywrightAutomation:
    """Advanced web automation using Playwright for ONI."""
    
    def __init__(self, headless: bool = True, browser_type: str = "chromium", 
                slow_mo: int = 0, screenshots_dir: Optional[str] = None):
        """
        Initialize Playwright automation.
        
        Args:
            headless: Whether to run browser in headless mode
            browser_type: Browser type ('chromium', 'firefox', or 'webkit')
            slow_mo: Slow down operations by specified milliseconds
            screenshots_dir: Directory to save screenshots (default: temporary directory)
        """
        if not HAS_PLAYWRIGHT:
            raise ImportError("Playwright is not installed. Please install it with 'pip install playwright' and run 'playwright install'.")
        
        self.headless = headless
        self.browser_type = browser_type
        self.slow_mo = slow_mo
        self.screenshots_dir = screenshots_dir or os.path.join(tempfile.gettempdir(), "oni_playwright_screenshots")
        
        # Create screenshots directory if it doesn't exist
        os.makedirs(self.screenshots_dir, exist_ok=True)
        
        # Initialize Playwright objects
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        
        # Track state
        self.is_initialized = False
        self.current_url = None
        
        logger.info(f"Playwright automation initialized with {browser_type} browser")
    
    async def initialize(self) -> bool:
        """Initialize Playwright and launch browser."""
        try:
            self.playwright = await async_playwright().start()
            
            # Select browser type
            if self.browser_type == "chromium":
                browser_instance = self.playwright.chromium
            elif self.browser_type == "firefox":
                browser_instance = self.playwright.firefox
            elif self.browser_type == "webkit":
                browser_instance = self.playwright.webkit
            else:
                raise ValueError(f"Unsupported browser type: {self.browser_type}")
            
            # Launch browser
            self.browser = await browser_instance.launch(
                headless=self.headless,
                slow_mo=self.slow_mo
            )
            
            # Create context
            self.context = await self.browser.new_context(
                viewport={"width": 1280, "height": 800},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
            
            # Create page
            self.page = await self.context.new_page()
            
            # Set up event handlers
            self.page.on("console", lambda msg: logger.debug(f"Console {msg.type}: {msg.text}"))
            self.page.on("pageerror", lambda err: logger.error(f"Page error: {err}"))
            
            self.is_initialized = True
            logger.info("Playwright initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Playwright: {e}")
            return False
    
    async def close(self) -> None:
        """Close browser and Playwright."""
        try:
            if self.browser:
                await self.browser.close()
            
            if self.playwright:
                await self.playwright.stop()
                
            self.is_initialized = False
            logger.info("Playwright closed")
            
        except Exception as e:
            logger.error(f"Error closing Playwright: {e}")
    
    async def navigate(self, url: str, wait_until: str = "load") -> Dict[str, Any]:
        """
        Navigate to a URL.
        
        Args:
            url: URL to navigate to
            wait_until: When to consider navigation complete ('load', 'domcontentloaded', 'networkidle')
            
        Returns:
            Dictionary with navigation result
        """
        if not self.is_initialized:
            return {"success": False, "error": "Playwright not initialized"}
        
        try:
            # Add http:// prefix if missing
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            
            # Navigate to URL
            response = await self.page.goto(url, wait_until=wait_until)
            
            # Update current URL
            self.current_url = self.page.url
            
            # Take screenshot
            screenshot_path = os.path.join(self.screenshots_dir, f"navigate_{int(time.time())}.png")
            await self.page.screenshot(path=screenshot_path)
            
            logger.info(f"Navigated to {url}")
            return {
                "success": True,
                "url": self.current_url,
                "status": response.status if response else None,
                "screenshot": screenshot_path
            }
            
        except Exception as e:
            logger.error(f"Navigation error: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_page_content(self) -> Dict[str, Any]:
        """
        Get the content of the current page.
        
        Returns:
            Dictionary with page content
        """
        if not self.is_initialized:
            return {"success": False, "error": "Playwright not initialized"}
        
        try:
            # Get page title
            title = await self.page.title()
            
            # Get page content
            content = await self.page.content()
            
            # Get page text
            text = await self.page.evaluate("() => document.body.innerText")
            
            # Take screenshot
            screenshot_path = os.path.join(self.screenshots_dir, f"content_{int(time.time())}.png")
            await self.page.screenshot(path=screenshot_path)
            
            return {
                "success": True,
                "url": self.page.url,
                "title": title,
                "content": content,
                "text": text,
                "screenshot": screenshot_path
            }
            
        except Exception as e:
            logger.error(f"Error getting page content: {e}")
            return {"success": False, "error": str(e)}
    
    async def click(self, selector: str, timeout: int = 5000) -> Dict[str, Any]:
        """
        Click an element on the page.
        
        Args:
            selector: CSS selector for the element
            timeout: Timeout in milliseconds
            
        Returns:
            Dictionary with click result
        """
        if not self.is_initialized:
            return {"success": False, "error": "Playwright not initialized"}
        
        try:
            # Wait for element to be visible
            await self.page.wait_for_selector(selector, state="visible", timeout=timeout)
            
            # Click element
            await self.page.click(selector)
            
            # Take screenshot
            screenshot_path = os.path.join(self.screenshots_dir, f"click_{int(time.time())}.png")
            await self.page.screenshot(path=screenshot_path)
            
            logger.info(f"Clicked element: {selector}")
            return {
                "success": True,
                "selector": selector,
                "screenshot": screenshot_path
            }
            
        except Exception as e:
            logger.error(f"Click error: {e}")
            return {"success": False, "error": str(e)}
    
    async def type_text(self, selector: str, text: str, delay: int = 50) -> Dict[str, Any]:
        """
        Type text into an input field.
        
        Args:
            selector: CSS selector for the input field
            text: Text to type
            delay: Delay between keystrokes in milliseconds
            
        Returns:
            Dictionary with typing result
        """
        if not self.is_initialized:
            return {"success": False, "error": "Playwright not initialized"}
        
        try:
            # Wait for element to be visible
            await self.page.wait_for_selector(selector, state="visible")
            
            # Click the element first to focus it
            await self.page.click(selector)
            
            # Clear existing text
            await self.page.evaluate(f"document.querySelector('{selector}').value = ''")
            
            # Type text
            await self.page.type(selector, text, delay=delay)
            
            # Take screenshot
            screenshot_path = os.path.join(self.screenshots_dir, f"type_{int(time.time())}.png")
            await self.page.screenshot(path=screenshot_path)
            
            logger.info(f"Typed text into {selector}: {text}")
            return {
                "success": True,
                "selector": selector,
                "text": text,
                "screenshot": screenshot_path
            }
            
        except Exception as e:
            logger.error(f"Type error: {e}")
            return {"success": False, "error": str(e)}
    
    async def select_option(self, selector: str, value: str) -> Dict[str, Any]:
        """
        Select an option from a dropdown.
        
        Args:
            selector: CSS selector for the select element
            value: Value to select
            
        Returns:
            Dictionary with selection result
        """
        if not self.is_initialized:
            return {"success": False, "error": "Playwright not initialized"}
        
        try:
            # Wait for element to be visible
            await self.page.wait_for_selector(selector, state="visible")
            
            # Select option
            await self.page.select_option(selector, value)
            
            # Take screenshot
            screenshot_path = os.path.join(self.screenshots_dir, f"select_{int(time.time())}.png")
            await self.page.screenshot(path=screenshot_path)
            
            logger.info(f"Selected option in {selector}: {value}")
            return {
                "success": True,
                "selector": selector,
                "value": value,
                "screenshot": screenshot_path
            }
            
        except Exception as e:
            logger.error(f"Select error: {e}")
            return {"success": False, "error": str(e)}
    
    async def extract_data(self, selectors: Dict[str, str]) -> Dict[str, Any]:
        """
        Extract data from the page using multiple selectors.
        
        Args:
            selectors: Dictionary mapping data keys to CSS selectors
            
        Returns:
            Dictionary with extracted data
        """
        if not self.is_initialized:
            return {"success": False, "error": "Playwright not initialized"}
        
        try:
            data = {}
            
            for key, selector in selectors.items():
                try:
                    # Wait for element to be visible
                    await self.page.wait_for_selector(selector, state="visible", timeout=5000)
                    
                    # Get element text
                    element = await self.page.query_selector(selector)
                    if element:
                        text = await element.inner_text()
                        data[key] = text.strip()
                    else:
                        data[key] = None
                        
                except Exception as e:
                    logger.warning(f"Failed to extract {key} using selector {selector}: {e}")
                    data[key] = None
            
            # Take screenshot
            screenshot_path = os.path.join(self.screenshots_dir, f"extract_{int(time.time())}.png")
            await self.page.screenshot(path=screenshot_path)
            
            logger.info(f"Extracted data from {len(selectors)} selectors")
            return {
                "success": True,
                "data": data,
                "screenshot": screenshot_path
            }
            
        except Exception as e:
            logger.error(f"Data extraction error: {e}")
            return {"success": False, "error": str(e)}
    
    async def wait_for_navigation(self, timeout: int = 30000, wait_until: str = "load") -> Dict[str, Any]:
        """
        Wait for navigation to complete.
        
        Args:
            timeout: Timeout in milliseconds
            wait_until: When to consider navigation complete ('load', 'domcontentloaded', 'networkidle')
            
        Returns:
            Dictionary with navigation result
        """
        if not self.is_initialized:
            return {"success": False, "error": "Playwright not initialized"}
        
        try:
            # Wait for navigation
            response = await self.page.wait_for_navigation(timeout=timeout, wait_until=wait_until)
            
            # Update current URL
            self.current_url = self.page.url
            
            # Take screenshot
            screenshot_path = os.path.join(self.screenshots_dir, f"navigation_{int(time.time())}.png")
            await self.page.screenshot(path=screenshot_path)
            
            logger.info(f"Navigation complete: {self.current_url}")
            return {
                "success": True,
                "url": self.current_url,
                "status": response.status if response else None,
                "screenshot": screenshot_path
            }
            
        except Exception as e:
            logger.error(f"Navigation wait error: {e}")
            return {"success": False, "error": str(e)}
    
    async def execute_javascript(self, script: str) -> Dict[str, Any]:
        """
        Execute JavaScript on the page.
        
        Args:
            script: JavaScript code to execute
            
        Returns:
            Dictionary with execution result
        """
        if not self.is_initialized:
            return {"success": False, "error": "Playwright not initialized"}
        
        try:
            # Execute script
            result = await self.page.evaluate(script)
            
            # Take screenshot
            screenshot_path = os.path.join(self.screenshots_dir, f"js_{int(time.time())}.png")
            await self.page.screenshot(path=screenshot_path)
            
            logger.info(f"Executed JavaScript: {script[:50]}...")
            return {
                "success": True,
                "result": result,
                "screenshot": screenshot_path
            }
            
        except Exception as e:
            logger.error(f"JavaScript execution error: {e}")
            return {"success": False, "error": str(e)}
    
    async def fill_form(self, form_data: Dict[str, str], submit_selector: Optional[str] = None) -> Dict[str, Any]:
        """
        Fill a form with data.
        
        Args:
            form_data: Dictionary mapping field selectors to values
            submit_selector: CSS selector for the submit button (optional)
            
        Returns:
            Dictionary with form submission result
        """
        if not self.is_initialized:
            return {"success": False, "error": "Playwright not initialized"}
        
        try:
            # Fill each field
            for selector, value in form_data.items():
                await self.page.fill(selector, value)
                logger.info(f"Filled {selector} with {value}")
            
            # Take screenshot after filling
            screenshot_before = os.path.join(self.screenshots_dir, f"form_before_{int(time.time())}.png")
            await self.page.screenshot(path=screenshot_before)
            
            # Submit form if selector provided
            if submit_selector:
                await self.page.click(submit_selector)
                
                # Wait for navigation
                try:
                    await self.page.wait_for_load_state("networkidle")
                except:
                    # Navigation might not happen, ignore errors
                    pass
                
                # Take screenshot after submission
                screenshot_after = os.path.join(self.screenshots_dir, f"form_after_{int(time.time())}.png")
                await self.page.screenshot(path=screenshot_after)
                
                logger.info(f"Submitted form using {submit_selector}")
                return {
                    "success": True,
                    "form_data": form_data,
                    "submitted": True,
                    "url": self.page.url,
                    "screenshot_before": screenshot_before,
                    "screenshot_after": screenshot_after
                }
            
            logger.info("Form filled but not submitted")
            return {
                "success": True,
                "form_data": form_data,
                "submitted": False,
                "screenshot": screenshot_before
            }
            
        except Exception as e:
            logger.error(f"Form fill error: {e}")
            return {"success": False, "error": str(e)}
    
    async def take_screenshot(self, selector: Optional[str] = None, 
                            full_page: bool = False) -> Dict[str, Any]:
        """
        Take a screenshot of the page or a specific element.
        
        Args:
            selector: CSS selector for the element to screenshot (optional)
            full_page: Whether to take a screenshot of the full page
            
        Returns:
            Dictionary with screenshot result
        """
        if not self.is_initialized:
            return {"success": False, "error": "Playwright not initialized"}
        
        try:
            screenshot_path = os.path.join(self.screenshots_dir, f"screenshot_{int(time.time())}.png")
            
            if selector:
                # Wait for element to be visible
                await self.page.wait_for_selector(selector, state="visible")
                
                # Take screenshot of element
                element = await self.page.query_selector(selector)
                await element.screenshot(path=screenshot_path)
                
                logger.info(f"Took screenshot of element {selector}")
                return {
                    "success": True,
                    "selector": selector,
                    "screenshot": screenshot_path
                }
            else:
                # Take screenshot of page
                await self.page.screenshot(path=screenshot_path, full_page=full_page)
                
                logger.info(f"Took screenshot of {'full page' if full_page else 'viewport'}")
                return {
                    "success": True,
                    "full_page": full_page,
                    "screenshot": screenshot_path
                }
                
        except Exception as e:
            logger.error(f"Screenshot error: {e}")
            return {"success": False, "error": str(e)}
    
    async def scroll(self, selector: Optional[str] = None, 
                   position: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """
        Scroll the page or a specific element.
        
        Args:
            selector: CSS selector for the element to scroll (optional)
            position: Dictionary with x and y coordinates to scroll to (optional)
            
        Returns:
            Dictionary with scroll result
        """
        if not self.is_initialized:
            return {"success": False, "error": "Playwright not initialized"}
        
        try:
            if selector:
                # Wait for element to be visible
                await self.page.wait_for_selector(selector, state="visible")
                
                # Scroll element into view
                element = await self.page.query_selector(selector)
                await element.scroll_into_view_if_needed()
                
                logger.info(f"Scrolled element into view: {selector}")
                return {
                    "success": True,
                    "selector": selector
                }
            elif position:
                # Scroll to position
                x = position.get("x", 0)
                y = position.get("y", 0)
                
                await self.page.evaluate(f"window.scrollTo({x}, {y})")
                
                logger.info(f"Scrolled to position: ({x}, {y})")
                return {
                    "success": True,
                    "position": position
                }
            else:
                # Scroll to bottom of page
                await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                
                logger.info("Scrolled to bottom of page")
                return {
                    "success": True,
                    "position": "bottom"
                }
                
        except Exception as e:
            logger.error(f"Scroll error: {e}")
            return {"success": False, "error": str(e)}
    
    async def wait_for_selector(self, selector: str, state: str = "visible", 
                              timeout: int = 30000) -> Dict[str, Any]:
        """
        Wait for an element to appear on the page.
        
        Args:
            selector: CSS selector for the element
            state: State to wait for ('attached', 'detached', 'visible', 'hidden')
            timeout: Timeout in milliseconds
            
        Returns:
            Dictionary with wait result
        """
        if not self.is_initialized:
            return {"success": False, "error": "Playwright not initialized"}
        
        try:
            # Wait for element
            await self.page.wait_for_selector(selector, state=state, timeout=timeout)
            
            # Take screenshot
            screenshot_path = os.path.join(self.screenshots_dir, f"wait_{int(time.time())}.png")
            await self.page.screenshot(path=screenshot_path)
            
            logger.info(f"Element found: {selector}")
            return {
                "success": True,
                "selector": selector,
                "state": state,
                "screenshot": screenshot_path
            }
            
        except Exception as e:
            logger.error(f"Wait error: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_cookies(self) -> Dict[str, Any]:
        """
        Get cookies from the current page.
        
        Returns:
            Dictionary with cookies
        """
        if not self.is_initialized:
            return {"success": False, "error": "Playwright not initialized"}
        
        try:
            # Get cookies
            cookies = await self.context.cookies()
            
            logger.info(f"Got {len(cookies)} cookies")
            return {
                "success": True,
                "cookies": cookies
            }
            
        except Exception as e:
            logger.error(f"Cookie error: {e}")
            return {"success": False, "error": str(e)}
    
    async def set_cookies(self, cookies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Set cookies for the current context.
        
        Args:
            cookies: List of cookie objects
            
        Returns:
            Dictionary with operation result
        """
        if not self.is_initialized:
            return {"success": False, "error": "Playwright not initialized"}
        
        try:
            # Set cookies
            await self.context.add_cookies(cookies)
            
            logger.info(f"Set {len(cookies)} cookies")
            return {
                "success": True,
                "count": len(cookies)
            }
            
        except Exception as e:
            logger.error(f"Cookie error: {e}")
            return {"success": False, "error": str(e)}
    
    async def intercept_requests(self, url_pattern: str, handler_type: str = "block") -> Dict[str, Any]:
        """
        Intercept requests matching a pattern.
        
        Args:
            url_pattern: Regular expression pattern for URLs to intercept
            handler_type: Type of handler ('block', 'abort', 'continue')
            
        Returns:
            Dictionary with operation result
        """
        if not self.is_initialized:
            return {"success": False, "error": "Playwright not initialized"}
        
        try:
            # Create route handler
            async def route_handler(route, request):
                if handler_type == "block" or handler_type == "abort":
                    await route.abort()
                    logger.info(f"Blocked request: {request.url}")
                else:
                    await route.continue_()
                    logger.info(f"Continued request: {request.url}")
            
            # Add route
            await self.page.route(url_pattern, route_handler)
            
            logger.info(f"Set up request interception for pattern: {url_pattern}")
            return {
                "success": True,
                "url_pattern": url_pattern,
                "handler_type": handler_type
            }
            
        except Exception as e:
            logger.error(f"Interception error: {e}")
            return {"success": False, "error": str(e)}
    
    async def run_test(self, test_script: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run a test script consisting of multiple actions.
        
        Args:
            test_script: List of action dictionaries
            
        Returns:
            Dictionary with test results
        """
        if not self.is_initialized:
            return {"success": False, "error": "Playwright not initialized"}
        
        results = []
        success = True
        
        try:
            for i, action in enumerate(test_script):
                action_type = action.get("type")
                
                if not action_type:
                    logger.warning(f"Action {i} has no type, skipping")
                    continue
                
                logger.info(f"Executing action {i}: {action_type}")
                
                # Execute action based on type
                if action_type == "navigate":
                    result = await self.navigate(action.get("url"), action.get("wait_until", "load"))
                
                elif action_type == "click":
                    result = await self.click(action.get("selector"), action.get("timeout", 5000))
                
                elif action_type == "type":
                    result = await self.type_text(action.get("selector"), action.get("text"), action.get("delay", 50))
                
                elif action_type == "select":
                    result = await self.select_option(action.get("selector"), action.get("value"))
                
                elif action_type == "extract":
                    result = await self.extract_data(action.get("selectors", {}))
                
                elif action_type == "wait":
                    if "selector" in action:
                        result = await self.wait_for_selector(action.get("selector"), action.get("state", "visible"), action.get("timeout", 30000))
                    else:
                        # Wait for specified time
                        await asyncio.sleep(action.get("time", 1))
                        result = {"success": True, "time": action.get("time", 1)}
                
                elif action_type == "screenshot":
                    result = await self.take_screenshot(action.get("selector"), action.get("full_page", False))
                
                elif action_type == "scroll":
                    result = await self.scroll(action.get("selector"), action.get("position"))
                
                elif action_type == "js":
                    result = await self.execute_javascript(action.get("script", ""))
                
                elif action_type == "form":
                    result = await self.fill_form(action.get("data", {}), action.get("submit"))
                
                else:
                    logger.warning(f"Unknown action type: {action_type}")
                    result = {"success": False, "error": f"Unknown action type: {action_type}"}
                
                # Add result to results list
                result["action_type"] = action_type
                result["action_index"] = i
                results.append(result)
                
                # Stop on failure if specified
                if not result.get("success", False) and action.get("stop_on_failure", False):
                    logger.warning(f"Stopping test script due to failure at action {i}")
                    success = False
                    break
            
            logger.info(f"Test script completed with {len(results)} actions")
            return {
                "success": success,
                "results": results,
                "url": self.page.url if self.page else None
            }
            
        except Exception as e:
            logger.error(f"Test script error: {e}")
            return {"success": False, "error": str(e), "results": results}
    
    async def scrape_data(self, selectors: Dict[str, str], 
                        pagination_selector: Optional[str] = None, 
                        max_pages: int = 1) -> Dict[str, Any]:
        """
        Scrape data from multiple pages.
        
        Args:
            selectors: Dictionary mapping data keys to CSS selectors
            pagination_selector: CSS selector for the next page button (optional)
            max_pages: Maximum number of pages to scrape
            
        Returns:
            Dictionary with scraped data
        """
        if not self.is_initialized:
            return {"success": False, "error": "Playwright not initialized"}
        
        all_data = []
        pages_scraped = 0
        
        try:
            for page_num in range(max_pages):
                # Extract data from current page
                result = await self.extract_data(selectors)
                
                if not result.get("success", False):
                    logger.warning(f"Failed to extract data from page {page_num + 1}")
                    break
                
                # Add data to results
                all_data.append(result.get("data", {}))
                pages_scraped += 1
                
                # Take screenshot
                screenshot_path = os.path.join(self.screenshots_dir, f"scrape_page_{page_num + 1}.png")
                await self.page.screenshot(path=screenshot_path)
                
                # Check if we should navigate to next page
                if pagination_selector and page_num < max_pages - 1:
                    # Check if next page button exists
                    next_button = await self.page.query_selector(pagination_selector)
                    
                    if next_button:
                        # Click next page button
                        await next_button.click()
                        
                        # Wait for navigation
                        try:
                            await self.page.wait_for_load_state("networkidle")
                        except:
                            # Navigation might not happen, ignore errors
                            pass
                        
                        logger.info(f"Navigated to page {page_num + 2}")
                    else:
                        logger.info(f"No next page button found, stopping at page {page_num + 1}")
                        break
                
            logger.info(f"Scraped data from {pages_scraped} pages")
            return {
                "success": True,
                "data": all_data,
                "pages_scraped": pages_scraped
            }
            
        except Exception as e:
            logger.error(f"Scraping error: {e}")
            return {"success": False, "error": str(e), "data": all_data, "pages_scraped": pages_scraped}
    
    async def login(self, url: str, username_selector: str, password_selector: str,
                  submit_selector: str, username: str, password: str,
                  success_indicator: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform a login operation.
        
        Args:
            url: Login page URL
            username_selector: CSS selector for username field
            password_selector: CSS selector for password field
            submit_selector: CSS selector for submit button
            username: Username to enter
            password: Password to enter
            success_indicator: CSS selector that indicates successful login (optional)
            
        Returns:
            Dictionary with login result
        """
        if not self.is_initialized:
            return {"success": False, "error": "Playwright not initialized"}
        
        try:
            # Navigate to login page
            nav_result = await self.navigate(url)
            
            if not nav_result.get("success", False):
                return {"success": False, "error": f"Failed to navigate to login page: {nav_result.get('error')}"}
            
            # Fill login form
            form_data = {
                username_selector: username,
                password_selector: password
            }
            
            form_result = await self.fill_form(form_data, submit_selector)
            
            if not form_result.get("success", False):
                return {"success": False, "error": f"Failed to fill login form: {form_result.get('error')}"}
            
            # Wait for navigation
            try:
                await self.page.wait_for_load_state("networkidle")
            except:
                # Navigation might not happen, ignore errors
                pass
            
            # Take screenshot
            screenshot_path = os.path.join(self.screenshots_dir, f"login_{int(time.time())}.png")
            await self.page.screenshot(path=screenshot_path)
            
            # Check for success indicator if provided
            if success_indicator:
                try:
                    await self.page.wait_for_selector(success_indicator, state="visible", timeout=5000)
                    login_success = True
                except:
                    login_success = False
            else:
                # Assume success if no indicator provided
                login_success = True
            
            # Get cookies
            cookies = await self.context.cookies()
            
            logger.info(f"Login {'successful' if login_success else 'failed'}")
            return {
                "success": login_success,
                "url": self.page.url,
                "cookies": cookies,
                "screenshot": screenshot_path
            }
            
        except Exception as e:
            logger.error(f"Login error: {e}")
            return {"success": False, "error": str(e)}
    
    async def download_file(self, url: str, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Download a file from a URL.
        
        Args:
            url: URL of the file to download
            save_path: Path to save the file (optional)
            
        Returns:
            Dictionary with download result
        """
        if not self.is_initialized:
            return {"success": False, "error": "Playwright not initialized"}
        
        try:
            # Set up download handler
            download_path = save_path or os.path.join(tempfile.gettempdir(), f"download_{int(time.time())}")
            
            # Create download directory if it doesn't exist
            os.makedirs(os.path.dirname(download_path), exist_ok=True)
            
            # Navigate to URL
            await self.navigate(url)
            
            # Set up download listener
            async with self.page.expect_download() as download_info:
                # Trigger download (click a link or button)
                await self.page.click("a[href*='download'], a[download], button[download]")
            
            # Get download
            download = await download_info.value
            
            # Save file
            await download.save_as(download_path)
            
            logger.info(f"Downloaded file to {download_path}")
            return {
                "success": True,
                "url": url,
                "path": download_path,
                "filename": download.suggested_filename
            }
            
        except Exception as e:
            logger.error(f"Download error: {e}")
            return {"success": False, "error": str(e)}
    
    async def upload_file(self, selector: str, file_path: str) -> Dict[str, Any]:
        """
        Upload a file.
        
        Args:
            selector: CSS selector for the file input
            file_path: Path to the file to upload
            
        Returns:
            Dictionary with upload result
        """
        if not self.is_initialized:
            return {"success": False, "error": "Playwright not initialized"}
        
        try:
            # Wait for element to be visible
            await self.page.wait_for_selector(selector)
            
            # Upload file
            await self.page.set_input_files(selector, file_path)
            
            # Take screenshot
            screenshot_path = os.path.join(self.screenshots_dir, f"upload_{int(time.time())}.png")
            await self.page.screenshot(path=screenshot_path)
            
            logger.info(f"Uploaded file {file_path} using {selector}")
            return {
                "success": True,
                "selector": selector,
                "file_path": file_path,
                "screenshot": screenshot_path
            }
            
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_network_requests(self) -> Dict[str, Any]:
        """
        Get network requests made by the page.
        
        Returns:
            Dictionary with network requests
        """
        if not self.is_initialized:
            return {"success": False, "error": "Playwright not initialized"}
        
        try:
            # Get network requests using JavaScript
            requests = await self.page.evaluate("""
                () => {
                    const entries = performance.getEntriesByType('resource');
                    return entries.map(entry => ({
                        name: entry.name,
                        initiatorType: entry.initiatorType,
                        duration: entry.duration,
                        size: entry.transferSize
                    }));
                }
            """)
            
            logger.info(f"Got {len(requests)} network requests")
            return {
                "success": True,
                "requests": requests
            }
            
        except Exception as e:
            logger.error(f"Network request error: {e}")
            return {"success": False, "error": str(e)}
    
    async def run_accessibility_audit(self) -> Dict[str, Any]:
        """
        Run an accessibility audit on the current page.
        
        Returns:
            Dictionary with accessibility audit results
        """
        if not self.is_initialized:
            return {"success": False, "error": "Playwright not initialized"}
        
        try:
            # Run axe-core for accessibility testing
            audit_results = await self.page.evaluate("""
                async () => {
                    // Load axe-core from CDN
                    const script = document.createElement('script');
                    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.4.2/axe.min.js';
                    document.head.appendChild(script);
                    
                    // Wait for script to load
                    await new Promise(resolve => script.onload = resolve);
                    
                    // Run axe
                    return await axe.run();
                }
            """)
            
            logger.info(f"Completed accessibility audit with {len(audit_results.get('violations', []))} violations")
            return {
                "success": True,
                "results": audit_results
            }
            
        except Exception as e:
            logger.error(f"Accessibility audit error: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the current page.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.is_initialized:
            return {"success": False, "error": "Playwright not initialized"}
        
        try:
            # Get performance metrics using JavaScript
            metrics = await self.page.evaluate("""
                () => {
                    const perfEntries = performance.getEntriesByType('navigation')[0];
                    return {
                        domContentLoaded: perfEntries.domContentLoadedEventEnd - perfEntries.domContentLoadedEventStart,
                        load: perfEntries.loadEventEnd - perfEntries.loadEventStart,
                        domInteractive: perfEntries.domInteractive - perfEntries.startTime,
                        firstPaint: performance.getEntriesByName('first-paint')[0]?.startTime,
                        firstContentfulPaint: performance.getEntriesByName('first-contentful-paint')[0]?.startTime,
                        totalTime: perfEntries.duration
                    };
                }
            """)
            
            logger.info(f"Got performance metrics: {metrics}")
            return {
                "success": True,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Performance metrics error: {e}")
            return {"success": False, "error": str(e)}
    
    async def run_web_automation(self, script: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run a web automation script.
        
        Args:
            script: List of actions to perform
            
        Returns:
            Dictionary with automation results
        """
        # Initialize Playwright if not already initialized
        if not self.is_initialized:
            init_result = await self.initialize()
            if not init_result:
                return {"success": False, "error": "Failed to initialize Playwright"}
        
        try:
            # Run test script
            result = await self.run_test(script)
            
            return result
            
        except Exception as e:
            logger.error(f"Web automation error: {e}")
            return {"success": False, "error": str(e)}
        finally:
            # Close Playwright
            await self.close()

# Example usage
async def main():
    # Create automation instance
    automation = PlaywrightAutomation(headless=False)
    
    # Initialize
    await automation.initialize()
    
    try:
        # Navigate to a website
        await automation.navigate("https://example.com")
        
        # Extract data
        data = await automation.extract_data({
            "title": "h1",
            "description": "p"
        })
        
        print(f"Extracted data: {data}")
        
        # Run a test script
        test_script = [
            {"type": "navigate", "url": "https://example.com"},
            {"type": "click", "selector": "a"},
            {"type": "wait", "time": 2},
            {"type": "screenshot", "full_page": True}
        ]
        
        result = await automation.run_test(test_script)
        print(f"Test result: {result}")
        
    finally:
        # Close automation
        await automation.close()

if __name__ == "__main__":
    asyncio.run(main())