#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit and maintain technical_report.md against the codebase.

Usage:
    python scripts/maintain_technical_report.py [--fix] [--verbose]
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
REPORT_PATH = REPO_ROOT / "docs" / "technical_report.md"
# Canonical source-of-truth arch (renamed 2026-06-14 from rcc_bowtie).
ARCH_NEURO = REPO_ROOT / "architectures" / "master" / "arch.neuro"
ARCHIVE_DIR = REPO_ROOT / "docs" / "archive"


class ReportAuditor:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.issues = []
        self.fixes = []

    def audit(self):
        """Run full audit."""
        if not REPORT_PATH.exists():
            print("ERROR: {} not found".format(REPORT_PATH))
            return 1

        print("=" * 70)
        print("TECHNICAL REPORT AUDIT")
        print("=" * 70)

        self._audit_arch_neuro_drift()
        self._audit_stale_files()
        self._report_findings()

        return 1 if self.issues else 0

    def _audit_arch_neuro_drift(self):
        """Check for drift between arch.neuro and report."""
        print("\n[Phase 1] Architecture Configuration Drift")

        if not ARCH_NEURO.exists():
            self.issues.append(("ERROR", str(ARCH_NEURO) + " not found"))
            return

        with open(ARCH_NEURO, encoding='utf-8') as f:
            arch_content = f.read()

        with open(REPORT_PATH, encoding='utf-8') as f:
            report_content = f.read()

        # Extract key presets
        presets = self._extract_presets(arch_content)
        for preset_name in presets:
            if preset_name not in report_content:
                self.issues.append(
                    ("WARN", "Preset '{}' in arch.neuro but not in report".format(preset_name))
                )
            elif self.verbose:
                print("  [OK] Preset '{}' documented".format(preset_name))

        # Check key hyperparameters
        key_hparams = [
            ("loss_clipping", "3.0"),
            ("dropout", "0.12"),
            ("pct_trunk", "0.4"),
        ]

        for hparam, _ in key_hparams:
            pattern = "{}:.*?[0-9.]+".format(hparam)
            if not re.search(pattern, arch_content):
                self.issues.append(("WARN", "Hyperparameter '{}' not in arch.neuro".format(hparam)))
            elif hparam in report_content:
                if self.verbose:
                    print("  [OK] Hyperparameter '{}' documented".format(hparam))
            else:
                self.issues.append(("WARN", "Hyperparameter '{}' in arch.neuro but not report".format(hparam)))

    def _audit_stale_files(self):
        """Identify stale documentation files."""
        print("\n[Phase 2] Stale Documentation Check")

        stale_files = []
        if (REPO_ROOT / "docs" / "OOD_PUSH_STAGES.md").exists():
            stale_files.append("docs/OOD_PUSH_STAGES.md")
            self.fixes.append("ARCHIVE: docs/OOD_PUSH_STAGES.md")

        if stale_files:
            self.issues.append(
                ("INFO", "Found {} stale doc files".format(len(stale_files)))
            )
            for f in stale_files:
                if self.verbose:
                    print("  -> {}".format(f))

    def _report_findings(self):
        """Print summary."""
        print("\n" + "=" * 70)
        print("AUDIT SUMMARY")
        print("=" * 70)

        errors = [i for i in self.issues if i[0] == "ERROR"]
        warnings = [i for i in self.issues if i[0] == "WARN"]
        infos = [i for i in self.issues if i[0] == "INFO"]

        if errors:
            print("\n[ERRORS] {} found:".format(len(errors)))
            for _, msg in errors:
                print("   {}".format(msg))

        if warnings:
            print("\n[WARNINGS] {} found:".format(len(warnings)))
            for _, msg in warnings:
                print("   {}".format(msg))

        if infos:
            print("\n[NOTICES]")
            for _, msg in infos:
                print("   {}".format(msg))

        if self.fixes:
            print("\n[RECOMMENDED FIXES]")
            for fix in self.fixes:
                print("   {}".format(fix))

        if not errors and not warnings:
            print("\n[PASS] Report audit passed. No issues detected.")
        else:
            print("\n[INFO] Run with --fix to apply auto-fixes.")

    @staticmethod
    def _extract_presets(arch_content):
        """Extract preset names from arch.neuro."""
        presets = {}
        scales_match = re.search(r'scales:\s*\{([^}]+)\}', arch_content, re.DOTALL)
        if not scales_match:
            return presets

        scales_block = scales_match.group(1)
        preset_pattern = r'(\w+):\s*\{'
        for match in re.finditer(preset_pattern, scales_block):
            preset_name = match.group(1)
            if preset_name not in ("hardware", "default"):
                presets[preset_name] = {}

        return presets

    def apply_fixes(self):
        """Apply automated fixes."""
        print("\n[Applying Fixes]")

        if not ARCHIVE_DIR.exists():
            ARCHIVE_DIR.mkdir(parents=True)
            print("Created {}".format(ARCHIVE_DIR))

        old_file = REPO_ROOT / "docs" / "OOD_PUSH_STAGES.md"
        if old_file.exists():
            timestamp = datetime.now().strftime("%Y-%m-%d")
            archived_name = "{}_OOD_PUSH_STAGES.md".format(timestamp)
            archived_path = ARCHIVE_DIR / archived_name
            old_file.rename(archived_path)
            print("[OK] Archived: {}".format(archived_path.relative_to(REPO_ROOT)))


def main():
    parser = argparse.ArgumentParser(description="Audit technical_report.md")
    parser.add_argument("--fix", action="store_true", help="Apply auto-fixes")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    auditor = ReportAuditor(verbose=args.verbose)
    exit_code = auditor.audit()

    if args.fix:
        auditor.apply_fixes()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
