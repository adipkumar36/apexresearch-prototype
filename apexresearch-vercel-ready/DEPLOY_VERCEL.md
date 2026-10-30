# ApexResearch AI — free Vercel demo

This copy is adapted from the supplied prototype ZIP for Vercel's FastAPI/Python runtime. The page calls the API on the deployed site automatically; local development still uses `http://localhost:8000`.

## Deploy with Vercel CLI (Windows PowerShell)

1. Extract this ZIP to a normal folder, for example `C:\Users\lenovo\Documents\apexresearch-vercel`.
2. Open PowerShell in that extracted folder. You should see `main.py` and `vercel.json` when you run `dir`.
3. Install Node.js if `node --version` does not work.
4. Run:

```powershell
npx vercel
```

5. On first use, the CLI opens Vercel sign-in in a browser. Sign in, then accept the prompts to create/link a project and deploy a preview.
6. To publish the production URL, run:

```powershell
npx vercel --prod
```

The CLI prints the live URL. Open it in a browser to use the prototype.

## Included free-tier adjustments

- Smaller serverless dependencies: Sentence Transformers, FAISS, OCR, and the Anthropic SDK are omitted. Similarity uses the built-in hashing fallback, so results are a rough demo only and **must not be used to judge plagiarism or make academic decisions**. Multilingual detection can label text, but cross-language semantic matching is reduced without the multilingual model.
- Vercel's writable `/tmp` is used for SQLite. Accounts and reviewer reports are temporary and can disappear after a cold start or be isolated between serverless instances. Do not use real/private submissions or rely on reviewer sync in this demo.
- Large document uploads may exceed the hosting platform's request limits. Try a short text or small PDF first.
- Live arXiv/OpenAlex lookups require outbound requests to succeed and may be slow or rate-limited.

For persistent reviewer accounts and reports, connect a managed database and update the app to use it; that needs a database provider account. For stronger multilingual analysis, use a hosted embeddings API or a separate persistent model service.
