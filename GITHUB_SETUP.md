# Pushing to GitHub

## Step 1 — Create the repository on GitHub

1. Go to https://github.com/new
2. Repository name: `molecular-design-vae`
3. Visibility: **Public**
4. Do NOT initialise with README, .gitignore, or licence — we have those already
5. Click **Create repository**

## Step 2 — Initialise git locally

```bash
cd molecular-design-vae

git init
git add .
git commit -m "Initial commit: VAE-based molecular design pipeline"
```

## Step 3 — Push

```bash
git remote add origin https://github.com/Kaur-Simarpreet/molecular-design-vae.git
git branch -M main
git push -u origin main
```

Replace `Kaur-Simarpreet` with your GitHub username.

## Step 4 — Verify

Visit `https://github.com/Kaur-Simarpreet/molecular-design-vae` — the README should render with the comparison tables.

## Notes on large files

`vae.pt`, `tokenizer.pkl`, and `latents.pt` are excluded by `.gitignore` because they can be several hundred MB. Anyone cloning the repo just runs `python train_vae.py` to regenerate them locally.

If you want to share trained weights, consider using GitHub Releases or Hugging Face Hub to host the files.
