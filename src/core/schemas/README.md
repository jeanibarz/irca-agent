# Schemas Directory

## Overview

This directory contains JSON schemas defining the structure and validation rules for configuration data used within the project. These schemas are crucial for ensuring that the configuration data adhere to the expected format and maintain consistency throughout the application.

## Directory Structure

- The directory is organized into versioned subdirectories (e.g., `v1`, `v2`).
- Each subdirectory contains a `training_args.json` file, representing the schema for that specific version.

## Schema Versions

- Each version subdirectory represents a different iteration or version of the JSON schema.
- When updating or adding new features to the schema, increment the version accordingly and create a new subdirectory for the new version.

## Usage

- Ensure that any configurations in the project are validated against the appropriate version of the schema.
- Refer to these schemas for understanding the expected structure and constraints of configuration data.

## Contribution Guidelines

- When contributing new schemas or updating existing ones, adhere to the versioning system and directory structure.
- Ensure backward compatibility where possible, and create new versions for breaking changes.
