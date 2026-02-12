"""
Upload trained model to Hugging Face Hub
"""

from huggingface_hub import HfApi, create_repo, login

print("="*60)
print("UPLOAD MODEL TO HUGGING FACE")
print("="*60)

# Step 1: Login (you'll need your token)
print("\nStep 1: Login to Hugging Face")
print("Enter your HF token (from https://huggingface.co/settings/tokens)")
token = input("Token: ").strip()

login(token=token)
print("✓ Logged in successfully")

# Step 2: Enter your username and model name
print("\nStep 2: Model details")
username = input("Your HF username: ").strip()
model_name = input("Model name (default: news-bias-classifier): ").strip() or "news-bias-classifier"

repo_id = f"{username}/{model_name}"

# Step 3: Create repository
print(f"\nStep 3: Creating repository '{repo_id}'...")
try:
    create_repo(repo_id, token=token, repo_type="model", exist_ok=True)
    print("✓ Repository created")
except Exception as e:
    print(f"Note: {e}")
    print("(This is OK if repo already exists)")

# Step 4: Upload model files
print(f"\nStep 4: Uploading model files from ./bias-classifier-final/...")
print("This may take 2-5 minutes...")

api = HfApi()
api.upload_folder(
    folder_path="./bias-classifier-final",
    repo_id=repo_id,
    token=token,
)

print("\n" + "="*60)
print("✓ MODEL UPLOADED SUCCESSFULLY!")
print("="*60)
print(f"\nYour model is available at:")
print(f"https://huggingface.co/{repo_id}")
print(f"\nAPI URL (for your website):")
print(f"https://api-inference.huggingface.co/models/{repo_id}")
print("\nNext steps:")
print("1. Update index.html with the API URL above")
print("2. Add your HF token to index.html (line 216)")
print("3. Push to GitHub and enable Pages")
print("="*60)
