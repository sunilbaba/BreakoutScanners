import requests
import pandas as pd
import io
import os

def download_and_save_csv():
    print("--- STARTING NIFTY 500 LIST UPDATE ---")
    
    # Official NSE Archive URL
    url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
    
    # Headers are CRITICAL. NSE blocks requests without a User-Agent.
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }
    
    try:
        print(f"Connecting to: {url}")
        session = requests.Session()
        response = session.get(url, headers=headers, timeout=20)
        
        if response.status_code == 200:
            print("Download Successful.")
            
            # Read CSV content
            csv_content = response.content.decode('utf-8')
            
            # Validate it's actually a CSV using Pandas
            df = pd.read_csv(io.StringIO(csv_content))
            
            # Basic validation
            if 'Symbol' in df.columns:
                print(f"Validated: Found {len(df)} symbols.")
                
                # Save to file
                filename = "ind_nifty500list.csv"
                df.to_csv(filename, index=False)
                print(f"Saved to {filename}")
            else:
                print("Error: CSV downloaded but 'Symbol' column missing.")
                exit(1) # Fail the action
        else:
            print(f"Failed to download. HTTP Status: {response.status_code}")
            exit(1) # Fail the action
            
    except Exception as e:
        print(f"Critical Error: {e}")
        exit(1)

if __name__ == "__main__":
    download_and_save_csv()
