# Value Cell Mobile

Value Cell Mobile is a standalone **Expo + React Native** application for Android and iOS.
It consumes the existing SaaS REST API; it does not embed the desktop Web application or expose
exchange API secrets to the client.

## Why Expo + React Native

- Produces genuine Android and iOS applications from one TypeScript codebase.
- Reuses the SaaS authentication, tenant, strategy, market-data, paper and OKX Demo APIs.
- Stores only the SaaS access token in the platform secure store (`expo-secure-store`).
- Uses native navigation, touch controls, keyboard handling and an SVG K-line renderer rather than
  wrapping the existing desktop page in a WebView.

## Mobile capabilities

- Secure SaaS login and encrypted local session persistence.
- Tenant workspace switching and access-status display.
- Strategy workspace: list, create, configure, save, start and stop strategies.
- Paper and OKX Demo execution controls with server-enforced execution-target rules.
- 50+ strategy observation symbols: search, select and inspect one K-line at a time.
- Native candlestick chart with MA5 and MA20 overlays, interval and history-range controls.
- Mobile safety boundary: real trading stays separately gated by the existing desktop-side connection,
  risk policy, strategy binding and runtime authorization flow.

## Configure the API endpoint

```bash
cd mobile
cp .env.example .env
```

Set a public HTTPS API endpoint:

```env
EXPO_PUBLIC_API_BASE_URL=https://your-valuecell-domain.example/api/v1
```

For Android emulators, `localhost` is the emulator itself; use the host LAN address or deployed
HTTPS domain. Do not place exchange keys, JWT secrets, credential master keys or provider tokens in
this file.

## Development

```bash
cd mobile
bun install
bun run android
# macOS only: bun run ios
# browser layout preview: bun run web
```

## Validation and local Android builds

```bash
bun run typecheck
bun run export:android
```

`export:android` validates and produces a JavaScript bundle; it does not produce an APK/AAB.
For an installable Android artifact without consuming Expo EAS cloud-build quota, generate
the native project once, then build it locally with an installed Android SDK and JDK:

```bash
npx expo prebuild --platform android
cd android
./gradlew.bat assembleRelease
```

The release APK is written under
`android/app/build/outputs/apk/release/`. Do not commit the generated `android/` directory
or build artifacts unless the project intentionally adopts native-project ownership.

`eas.json` remains available for development, internal-preview and production cloud profiles.
Before a store release, set the final Android package and iOS bundle identifier in `app.json`,
configure signing, and run a physical-device test against the production API.
