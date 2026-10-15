"""
=============================================================================
FRAUDSHIELD AI // LOCAL DEVELOPMENT SERVER & SESSION 2 BACKEND PREPARATION
=============================================================================
This script provides:
1. Instant zero-dependency local web server for the FraudShield AI Frontend.
2. Scaffolding for Session 2: Python AI Backend (Logistic Regression / Bayes).

Usage:
    python app.py
    Navigate to: http://localhost:5000
=============================================================================
"""

import http.server
import socketserver
import os
import webbrowser
import sys

PORT = 5000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class CyberFrontendHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # Enable CORS and caching headers for high performance
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()

    def log_message(self, format, *args):
        # Clean terminal logging matching the cyber aesthetic
        sys.stdout.write(f"[FRAUDSHIELD HTTP] - {self.log_date_time_string()} - {format % args}\n")

def run_server():
    os.chdir(DIRECTORY)
    print("=" * 70)
    print("  FRAUDSHIELD AI // AUTONOMOUS CYBER DEFENSE CONSOLE")
    print("  Theme: Robotics + Futuristic HUD Interface")
    print("=" * 70)
    print(f"[*] Serving files from: {DIRECTORY}")
    print(f"[*] Core URL: http://localhost:{PORT}")
    print(f"[*] Session 1: Frontend Client Active (Sample & Heuristic Data)")
    print(f"[*] Session 2: Python AI Backend Scaffolding Ready")
    print("=" * 70)
    print("[*] Press Ctrl+C to terminate sentinel server.\n")

    # Allow port reuse to avoid 'Address already in use' errors
    socketserver.TCPServer.allow_reuse_address = True

    try:
        with socketserver.TCPServer(("", PORT), CyberFrontendHandler) as httpd:
            print(f"[ONLINE] Local Sentinel Server listening on port {PORT}...")
            # Automatically attempt to launch default browser
            try:
                webbrowser.open(f"http://localhost:{PORT}")
            except Exception:
                pass
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[OFFLINE] FraudShield AI Server terminated by user.")
    except Exception as e:
        print(f"\n[ERROR] Server failed on port {PORT}: {e}")
        print("Tip: You can also double click 'index.html' to open directly in any browser.")

if __name__ == "__main__":
    run_server()
