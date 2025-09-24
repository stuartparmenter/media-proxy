# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.2] - 2025-09-23

### Fixed
- Fix race condition in non-looping image packet transmission where tasks could exit before all UDP packets (including still frame resends) were fully sent

## [0.3.1] - 2025-09-23

### Added
- Add AppArmor security profile for Home Assistant add-on

### Fixed
- Fix exception handling

## [0.3.0] - 2025-09-23

### Changed
- Changed from sleeping a fixed delay after each frame to tracking absolute target frame times, preventing timing drift

## [0.2.9] - 2025-09-21

### Added
- Add color correction for files with ICC profiles in them
- Optimize background color fetching to only do once

### Changed
- For smaller images, load them in to memory rather than saving them to a tempfile

## [0.2.8] - 2025-09-21

### Added
- Add proper disposal handling for GIFs and APNGs

## [0.2.7] - 2025-09-21

### Fixed
- Force lines to flush after each newline
- Fix can_handle calls to look at the path but not query params
- Add a user agent to avoid some bot blockers and better handle 403s and other errors

### Changed
- Blend all transparent images against a black background

## [0.2.6] - 2025-09-21

### Fixed
- Fix webp framerate issue

## [0.2.5] - 2025-09-21

### Changed
- Improve how image resampling works

### Fixed
- Fix animated GIF performance issue

## [0.2.4] - 2025-09-17

### Changed
- Only include intel driver on x86
- Switch from abandoned jitterbit/get-changed-files to tj-actions/changed-files
- Bump home-assistant/builder from 2025.03.0 to 2025.09.0

### Fixed
- Fix linter issues

## [0.2.3] - 2025-09-17

### Added
- Add github workflows to build container packages

### Changed
- Remove unnecessary defaults on config.get calls since they already have a default value set

### Fixed
- Restore missing log configuration options from legacy server

## [0.2.0] - 2025-09-17

### Added
- Refactor media processing to protocol-based architecture
- Enable color range expansion by default
- Always prefer hardware acceleration when available
- Add development file exclusions to gitignore

### Changed
- Update README to reflect refactoring & bugfixes
- Reduce metrics logging frequency to minimize console spam
- Refactor control field management
- Migrate from imageio to PIL for image processing
- Refactor server.py in to separate files
- Handle youtube url resolution async
- Disable gamma correction by default

## [0.1.0] - 2025-09-16

### Added
- Change default config path to /config/media-proxy/config.yaml
- Use uvloop on non-Windows platforms
- Add README
- Add home assistant add-on files

### Changed
- Use rounding instead of flooring when converting from rgb888 to rgb565 and add slight shadow lift

### Fixed
- Initial release from https://github.com/stuartparmenter/lvgl-ddp-stream

[unreleased]: https://github.com/stuartparmenter/media-proxy/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/stuartparmenter/media-proxy/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/stuartparmenter/media-proxy/compare/v0.2.9...v0.3.0
[0.2.9]: https://github.com/stuartparmenter/media-proxy/compare/v0.1.0...v0.2.9
[0.2.8]: https://github.com/stuartparmenter/media-proxy/compare/v0.1.0...v0.2.8
[0.2.7]: https://github.com/stuartparmenter/media-proxy/compare/v0.1.0...v0.2.7
[0.2.6]: https://github.com/stuartparmenter/media-proxy/compare/v0.1.0...v0.2.6
[0.2.5]: https://github.com/stuartparmenter/media-proxy/compare/v0.1.0...v0.2.5
[0.2.4]: https://github.com/stuartparmenter/media-proxy/compare/v0.1.0...v0.2.4
[0.2.3]: https://github.com/stuartparmenter/media-proxy/compare/v0.1.0...v0.2.3
[0.2.0]: https://github.com/stuartparmenter/media-proxy/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/stuartparmenter/media-proxy/releases/tag/v0.1.0