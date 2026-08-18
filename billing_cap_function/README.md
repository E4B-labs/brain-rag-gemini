# Billing cap function

This Cloud Functions 2nd gen handler follows Google's official
"Disable billing when budget exceeded" pattern. A real budget notification at
or above 100% removes the billing account from `GOOGLE_CLOUD_PROJECT`.

For a safe smoke test, publish a notification payload containing
`"dryRun": true`; it logs the intended action without changing billing.
