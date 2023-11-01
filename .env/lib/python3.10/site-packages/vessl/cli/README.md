# Vessl CLI

## Development Guide

1. Set `ENABLE_SENTRY=false` for not notifying to sentry.

2. Use not default `print` but `print_(info|warning|error|success)` in vessl.cli.\_util

- This is for maintaining same styles for each print context.

3. Don't directly use `click.promt|confirm` or `inquiry.Prompt`, use what's defined on `vessl.cli._util`.

- This is also for the style consistency.

4. Changing `API_HOST` to `api.dev.vssl.ai`(dev api) doesn't work in cluster creation, because it seems that using dev-api on cluster-agent is not properly configured.
