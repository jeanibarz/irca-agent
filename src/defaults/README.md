# Defaults Directory

## Overview

This directory contains the default configuration values corresponding to each version of the JSON schemas found in the `schemas` directory. These default values are utilized for initializing configurations and providing standard fallbacks.

## Directory Structure

- Organized similarly to the `schemas` directory, the `defaults` directory includes versioned subdirectories.
- Each versioned subdirectory houses a `training_args.json` file with default settings that align with the corresponding schema version.

## Default Configuration Versions

- Each subdirectory version correlates directly with the schema version in the `schemas` directory.
- Default configurations in each version are tailored to match and work seamlessly with their respective schema versions.

## Usage

- Use these default configurations as a baseline or a reference point for setting up or modifying configurations in the project.
- These defaults ensure a consistent starting point for configuration values across different instances or environments.

## Contribution Guidelines

- When modifying default values, ensure alignment with the corresponding schema version in the `schemas` directory.
- Create new versions for significant changes to default configurations, especially when they introduce breaking changes or major shifts from the previous version.
