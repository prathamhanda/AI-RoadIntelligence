#!/usr/bin/env python3
"""
🛡️ Violence Detection Setup for Traffic Analysis System
Complete setup and configuration tool for Sightengine API integration

Features:
- Interactive credential setup
- Environment variable configuration
- API validation and testing
- Configuration file creation
- System verification

Author: Pratham Handa
GitHub: https://github.com/prathamhanda/IoT-Based_Traffic_Regulation
"""

import json
import os
import sys
from getpass import getpass

def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"🛡️ {title}")
    print("=" * 60)

def test_api_connection(api_user, api_secret):
    """Test the Sightengine API connection"""
    try:
        import requests
        import cv2
        import numpy as np
        
        print("\n🧪 Testing API connection...")
        
        # Create test image
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        cv2.putText(test_image, "TEST", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', test_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # Prepare API request
        files = {'media': buffer.tobytes()}
        data = {
            'models': 'violence',
            'api_user': api_user,
            'api_secret': api_secret
        }
        
        # Make API request
        response = requests.post(
            'https://api.sightengine.com/1.0/check.json',
            files=files,
            data=data,
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API test successful!")
            if 'violence' in result:
                violence_score = result['violence'].get('prob', 0)
                print(f"   Violence score: {violence_score:.3f}")
            if 'request' in result:
                print(f"   Request ID: {result['request'].get('id', 'Unknown')}")
            return True
        else:
            print(f"❌ API test failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except ImportError as e:
        print(f"⚠️ Cannot test API - missing dependency: {e}")
        print("   Run: pip install requests opencv-python numpy")
        return None
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def setup_environment_variables(api_user, api_secret):
    """Setup environment variables for API credentials"""
    print("\n🔧 Setting up environment variables...")
    
    try:
        if os.name == 'nt':  # Windows
            os.system(f'setx SIGHTENGINE_API_USER "{api_user}"')
            os.system(f'setx SIGHTENGINE_API_SECRET "{api_secret}"')
            print("✅ Environment variables set for Windows")
            print("   Note: Restart terminal or IDE to use new variables")
            print("\nFor immediate use in current session, run:")
            print(f"   set SIGHTENGINE_API_USER={api_user}")
            print(f"   set SIGHTENGINE_API_SECRET={api_secret}")
        else:  # Unix-like systems
            # Try to add to shell configuration files
            home_dir = os.path.expanduser("~")
            shell_files = [
                os.path.join(home_dir, ".bashrc"),
                os.path.join(home_dir, ".zshrc"),
                os.path.join(home_dir, ".profile")
            ]
            
            env_lines = [
                f'export SIGHTENGINE_API_USER="{api_user}"',
                f'export SIGHTENGINE_API_SECRET="{api_secret}"'
            ]
            
            added_to_file = False
            for shell_file in shell_files:
                if os.path.exists(shell_file):
                    with open(shell_file, 'a') as f:
                        f.write('\n# Sightengine API credentials for violence detection\n')
                        for line in env_lines:
                            f.write(line + '\n')
                    print(f"✅ Added to {shell_file}")
                    added_to_file = True
                    break
            
            if not added_to_file:
                print("⚠️ Could not find shell configuration file")
                print("   Add these lines to your shell configuration:")
                for line in env_lines:
                    print(f"   {line}")
            
            print("\nFor immediate use, run:")
            for line in env_lines:
                print(f"   {line}")
        
        # Set for current session
        os.environ['SIGHTENGINE_API_USER'] = api_user
        os.environ['SIGHTENGINE_API_SECRET'] = api_secret
        print("✅ Variables set for current session")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting environment variables: {e}")
        return False

def create_config_file(api_user, api_secret, threshold, interval, save_evidence):
    """Create configuration file with settings"""
    print("\n📄 Creating configuration file...")
    
    try:
        config = {
            "SIGHTENGINE_API_USER": api_user,
            "SIGHTENGINE_API_SECRET": api_secret,
            "VIOLENCE_DETECTION_ENABLED": True,
            "VIOLENCE_THRESHOLD": threshold,
            "VIOLENCE_CHECK_INTERVAL": interval,
            "VIOLENCE_SAVE_EVIDENCE": save_evidence,
            "VIOLENCE_MODELS": ["violence", "gore", "weapon"],
            "VIOLENCE_ALERT_ENABLED": True,
            "VIOLENCE_LOG_ENABLED": True
        }
        
        with open("config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print("✅ Configuration saved to config.json")
        print(f"   Threshold: {threshold}")
        print(f"   Check interval: {interval} frames")
        print(f"   Save evidence: {save_evidence}")
        print("   Models: violence, gore, weapon")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating configuration file: {e}")
        return False

def interactive_setup():
    """Interactive setup for violence detection"""
    print_header("Interactive Violence Detection Setup")
    
    print("This script will configure violence detection for the traffic analysis system.")
    print("You'll need Sightengine API credentials. If you don't have them:")
    print("1. Visit https://sightengine.com/")
    print("2. Create a free account (2,000 free operations/month)")
    print("3. Get your API User ID and Secret from the dashboard")
    print()
    
    # Get API credentials
    api_user = input("Enter your Sightengine API User ID: ").strip()
    if not api_user:
        print("❌ API User ID is required")
        return False
    
    api_secret = getpass("Enter your Sightengine API Secret: ").strip()
    if not api_secret:
        print("❌ API Secret is required")
        return False
    
    print("\n⚙️ Configuration Options")
    print("=" * 30)
    
    # Get configuration options
    threshold = input("Violence detection threshold (0.1-1.0, default 0.7): ").strip()
    if not threshold:
        threshold = 0.7
    else:
        try:
            threshold = float(threshold)
            if not 0.1 <= threshold <= 1.0:
                print("⚠️ Invalid threshold, using default 0.7")
                threshold = 0.7
        except ValueError:
            print("⚠️ Invalid threshold, using default 0.7")
            threshold = 0.7
    
    interval = input("Check interval in frames (default 30): ").strip()
    if not interval:
        interval = 30
    else:
        try:
            interval = int(interval)
            if interval < 1:
                interval = 30
        except ValueError:
            interval = 30
    
    save_evidence = input("Save evidence frames? (y/N): ").strip().lower()
    save_evidence = save_evidence in ['y', 'yes', '1', 'true']
    
    # Test API first
    api_test_result = test_api_connection(api_user, api_secret)
    if api_test_result is False:
        print("\n❌ API test failed. Please check your credentials.")
        return False
    elif api_test_result is None:
        print("\n⚠️ Could not test API due to missing dependencies.")
        print("   Continuing with setup...")
    
    # Setup environment variables
    env_success = setup_environment_variables(api_user, api_secret)
    
    # Create config file
    config_success = create_config_file(api_user, api_secret, threshold, interval, save_evidence)
    
    # Final summary
    print_header("Setup Complete")
    
    if env_success and config_success:
        print("🎉 Violence detection setup completed successfully!")
        print("\n📋 Configuration Summary:")
        print(f"   API User ID: {api_user}")
        print(f"   API Secret: {'*' * len(api_secret)}")
        print(f"   Violence detection: ENABLED")
        print(f"   Threshold: {threshold} ({threshold*100:.0f}% confidence)")
        print(f"   Check every: {interval} frames")
        print(f"   Evidence saving: {save_evidence}")
        
        print("\n🚀 Next Steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Test the system: python system_tools.py --test-violence")
        print("3. Run traffic analysis: python traffic_analysis.py --source 0")
        print("4. Check documentation: VIOLENCE_DETECTION.md")
        
        return True
    else:
        print("❌ Setup completed with errors. Please check the configuration.")
        return False

def direct_setup_with_credentials():
    """Direct setup with hardcoded credentials (for testing)"""
    # Your provided credentials
    api_user = "1229875066"
    api_secret = "wiiuJVuneTkQqPaiFVBfxu3N77GoV3ry"
    
    print_header("Direct Setup with Provided Credentials")
    
    print(f"Using provided credentials:")
    print(f"   API User: {api_user}")
    print(f"   API Secret: {'*' * len(api_secret)}")
    
    # Test API
    api_test_result = test_api_connection(api_user, api_secret)
    if api_test_result is False:
        print("\n❌ API test failed with provided credentials.")
        return False
    
    # Setup with default values
    threshold = 0.7
    interval = 30
    save_evidence = True
    
    # Setup environment variables
    env_success = setup_environment_variables(api_user, api_secret)
    
    # Create config file
    config_success = create_config_file(api_user, api_secret, threshold, interval, save_evidence)
    
    # Final summary
    print_header("Direct Setup Complete")
    
    if env_success and config_success:
        print("🎉 Violence detection configured successfully!")
        print("\n📋 Configuration:")
        print(f"   Threshold: {threshold} (70% confidence)")
        print(f"   Check interval: {interval} frames")
        print(f"   Evidence saving: {save_evidence}")
        
        print("\n🚀 Ready to use:")
        print("   python traffic_analysis.py --source 0")
        
        return True
    else:
        print("❌ Setup failed. Please check the errors above.")
        return False

def check_current_setup():
    """Check current violence detection setup"""
    print_header("Current Setup Status")
    
    # Check environment variables
    api_user = os.getenv('SIGHTENGINE_API_USER')
    api_secret = os.getenv('SIGHTENGINE_API_SECRET')
    
    print("🔧 Environment Variables:")
    if api_user and api_secret:
        print(f"   ✅ SIGHTENGINE_API_USER: {api_user}")
        print(f"   ✅ SIGHTENGINE_API_SECRET: {'*' * len(api_secret)}")
    else:
        print("   ❌ Environment variables not found")
    
    # Check configuration file
    print("\n📄 Configuration File:")
    if os.path.exists("config.json"):
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
            
            print("   ✅ config.json found")
            print(f"   API User: {config.get('SIGHTENGINE_API_USER', 'Not set')}")
            print(f"   Threshold: {config.get('VIOLENCE_THRESHOLD', 'Not set')}")
            print(f"   Check Interval: {config.get('VIOLENCE_CHECK_INTERVAL', 'Not set')}")
            print(f"   Save Evidence: {config.get('VIOLENCE_SAVE_EVIDENCE', 'Not set')}")
            print(f"   Detection Enabled: {config.get('VIOLENCE_DETECTION_ENABLED', 'Not set')}")
            
            # Test API with config credentials
            config_api_user = config.get('SIGHTENGINE_API_USER')
            config_api_secret = config.get('SIGHTENGINE_API_SECRET')
            
            if config_api_user and config_api_secret:
                print("\n🧪 Testing API with config credentials...")
                test_result = test_api_connection(config_api_user, config_api_secret)
                if test_result:
                    print("   ✅ API working correctly")
                elif test_result is False:
                    print("   ❌ API test failed")
                else:
                    print("   ⚠️ Could not test API (missing dependencies)")
            
        except Exception as e:
            print(f"   ❌ Error reading config.json: {e}")
    else:
        print("   ❌ config.json not found")
    
    # Check if violence detector module exists
    print("\n🛡️ Violence Detection Module:")
    if os.path.exists("violence_detector.py"):
        print("   ✅ violence_detector.py found")
    else:
        print("   ❌ violence_detector.py missing")
    
    print("\n" + "=" * 60)

def main():
    """Main setup interface"""
    print("🛡️ Violence Detection Setup Tool")
    print("=" * 60)
    print("Configure violence detection for the IoT Traffic Analysis System")
    print()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--check":
            check_current_setup()
            return
        elif sys.argv[1] == "--direct":
            direct_setup_with_credentials()
            return
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python direct_setup.py           - Interactive setup")
            print("  python direct_setup.py --check   - Check current setup")
            print("  python direct_setup.py --direct  - Direct setup with hardcoded credentials")
            print("  python direct_setup.py --help    - Show this help")
            return
    
    # Default: interactive setup
    print("Choose setup method:")
    print("1. Interactive setup (enter your own credentials)")
    print("2. Direct setup (use hardcoded credentials)")
    print("3. Check current setup")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        interactive_setup()
    elif choice == "2":
        direct_setup_with_credentials()
    elif choice == "3":
        check_current_setup()
    elif choice == "4":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice. Please run again.")

if __name__ == "__main__":
    main()
