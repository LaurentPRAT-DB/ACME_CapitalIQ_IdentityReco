# Databricks Asset Bundles Migration Skill - Complete Index

## 📚 Complete File Reference

### Core Skill Files

| File | Purpose | Use When |
|------|---------|----------|
| **databricks-dabs-migration.md** | Main skill definition with all detection logic | Invoked automatically by `/databricks-dabs-migration` |
| **README.md** | User guide with examples and common issues | Need help using the skill |
| **DABS_MIGRATION_CHEATSHEET.md** | Quick reference for common problems | Need a quick solution |
| **MIGRATION_TEMPLATE.md** | Progress tracking template | Starting a new migration |
| **WORKFLOW_DIAGRAM.md** | Visual workflow diagrams | Understanding the process |
| **INDEX.md** | This file - complete reference | Finding the right resource |

---

## 🚀 Quick Start Guide

### 1. First Time Using the Skill?

**Read these in order:**
1. `SKILL_PACKAGE_SUMMARY.md` - Overview of what's included
2. `README.md` - How to use the skill
3. `DABS_MIGRATION_CHEATSHEET.md` - Quick reference

### 2. Ready to Migrate?

**Steps:**
1. Run: `/databricks-dabs-migration`
2. Review generated documentation (7 files)
3. Follow `MIGRATION_GUIDE.md`
4. Track progress in `MIGRATION_TEMPLATE.md`

### 3. Need Quick Help?

**Go to:**
- `DABS_MIGRATION_CHEATSHEET.md` for common errors
- `WORKFLOW_DIAGRAM.md` for visual process
- `README.md` troubleshooting section

---

## 📖 Documentation Categories

### Learning & Understanding

| Document | What You'll Learn |
|----------|-------------------|
| `SKILL_PACKAGE_SUMMARY.md` | What the skill does and why it exists |
| `README.md` | How to use the skill effectively |
| `WORKFLOW_DIAGRAM.md` | Visual understanding of the process |
| `DATABRICKS_PATH_ALIGNMENT.md`* | Real-world example from entity matching project |

*Located in project root

### Reference & Quick Lookup

| Document | What You'll Find |
|----------|------------------|
| `DABS_MIGRATION_CHEATSHEET.md` | Common errors and solutions |
| `INDEX.md` | This file - navigation guide |
| `databricks-dabs-migration.md` | Complete skill specification |

### Project Management

| Document | What You'll Use It For |
|----------|------------------------|
| `MIGRATION_TEMPLATE.md` | Track migration progress |
| Generated `MIGRATION_GUIDE.md` | Step-by-step instructions |
| Generated `DATABRICKS_MIGRATION_COMPLETE.md` | Final reference doc |

---

## 🎯 Use Case Directory

### Scenario 1: "I need to migrate my project to DABs"

**Path:**
1. Read: `SKILL_PACKAGE_SUMMARY.md`
2. Run: `/databricks-dabs-migration`
3. Copy: `MIGRATION_TEMPLATE.md` to track progress
4. Follow: Generated `MIGRATION_GUIDE.md`
5. Reference: `DABS_MIGRATION_CHEATSHEET.md` for issues

### Scenario 2: "I'm getting a specific error"

**Path:**
1. Check: `DABS_MIGRATION_CHEATSHEET.md` - Common Issues section
2. Search: `README.md` - Troubleshooting section
3. Run: `/databricks-dabs-migration --focus [category]`
4. Review: Generated fix documentation

### Scenario 3: "I want to understand the migration process"

**Path:**
1. Read: `WORKFLOW_DIAGRAM.md` - Visual overview
2. Read: `SKILL_PACKAGE_SUMMARY.md` - Detailed explanation
3. Study: `DATABRICKS_PATH_ALIGNMENT.md` - Real example
4. Try: `/databricks-dabs-migration --mode analysis` (no changes)

### Scenario 4: "I'm training my team on DABs migration"

**Path:**
1. Share: `SKILL_PACKAGE_SUMMARY.md` - Overview
2. Demo: `/databricks-dabs-migration` on sample project
3. Distribute: `DABS_MIGRATION_CHEATSHEET.md` as handout
4. Reference: `WORKFLOW_DIAGRAM.md` for discussions
5. Use: `MIGRATION_TEMPLATE.md` for their projects

### Scenario 5: "I need to validate an existing migration"

**Path:**
1. Run: `/databricks-dabs-migration --mode validate`
2. Review: Generated validation report
3. Fix: Any issues found
4. Document: In `MIGRATION_TEMPLATE.md`

---

## 🔍 Error Message Lookup

### "TypeError: 'type' object is not subscriptable"

**Solutions in:**
- `DABS_MIGRATION_CHEATSHEET.md` → Issue #1
- `README.md` → Common Issues → Type Annotation Compatibility
- `SKILL_PACKAGE_SUMMARY.md` → Issue 1

**Quick fix:** Add `from __future__ import annotations` to all `.py` files

### "OSError: Cannot save file into a non-existent directory"

**Solutions in:**
- `DABS_MIGRATION_CHEATSHEET.md` → Issue #3
- `README.md` → Common Issues → Directory Creation
- `SKILL_PACKAGE_SUMMARY.md` → Issue 4

**Quick fix:** Add `Path(filepath).parent.mkdir(parents=True, exist_ok=True)`

### "RestException: Could not find experiment with ID None"

**Solutions in:**
- `DABS_MIGRATION_CHEATSHEET.md` → Issue #4
- `README.md` → Common Issues → MLflow Experiment Setup
- `SKILL_PACKAGE_SUMMARY.md` → Issue 5

**Quick fix:** Add `mlflow.set_experiment()` before `mlflow.start_run()`

### "RESOURCE_DOES_NOT_EXIST: Parent directory ... does not exist"

**Solutions in:**
- `DABS_MIGRATION_CHEATSHEET.md` → Troubleshooting section
- `README.md` → MLflow Experiment Error

**Quick fix:** Use flat experiment path: `/Users/{username}/{catalog}-experiment`

---

## 📊 Skill Invocation Reference

### Full Migration
```bash
/databricks-dabs-migration
```

### Focused Analysis
```bash
/databricks-dabs-migration --focus paths        # Only path issues
/databricks-dabs-migration --focus dependencies # Only library issues
/databricks-dabs-migration --focus mlflow       # Only MLflow issues
/databricks-dabs-migration --focus config       # Only DABs YAML issues
```

### Different Modes
```bash
/databricks-dabs-migration --mode analysis  # No fixes, just analyze
/databricks-dabs-migration --mode plan      # Generate plan only
/databricks-dabs-migration --mode execute   # Analyze and fix (default)
/databricks-dabs-migration --mode validate  # Validate existing migration
```

### Output Control
```bash
/databricks-dabs-migration --output docs    # Generate documentation only
/databricks-dabs-migration --output code    # Generate code fixes only
/databricks-dabs-migration --output yaml    # Generate DABs configs only
/databricks-dabs-migration --output all     # Everything (default)
```

---

## 📂 Generated Files Reference

### After Running the Skill

The skill creates 7 documentation files:

| File | Phase | Content |
|------|-------|---------|
| `MIGRATION_DISCOVERY.md` | 1 | Codebase structure analysis |
| `PATH_MIGRATION_PLAN.md` | 2 | Path refactoring recommendations |
| `DEPENDENCY_MIGRATION_PLAN.md` | 3 | Library compatibility report |
| `CONFIG_REVIEW.md` | 4 | DABs configuration validation |
| `CODE_QUALITY_FIXES.md` | 5 | Code improvement recommendations |
| `MIGRATION_GUIDE.md` | 6 | Step-by-step migration instructions |
| `DATABRICKS_MIGRATION_COMPLETE.md` | 7 | Final reference documentation |

**Reading order:**
1. Start with `MIGRATION_GUIDE.md` (summary of all fixes needed)
2. Dive into specific plans as you work through each category
3. Use `DATABRICKS_MIGRATION_COMPLETE.md` as final reference

---

## 🛠️ Code Pattern Reference

### Where to Find Code Examples

| Pattern | Location |
|---------|----------|
| Path parameterization | `DABS_MIGRATION_CHEATSHEET.md` → Code Patterns |
| Safe file writing | `DABS_MIGRATION_CHEATSHEET.md` → Pattern: Safe File Writing |
| MLflow experiment setup | `DABS_MIGRATION_CHEATSHEET.md` → Pattern: MLflow Experiment Setup |
| Widget parameters | `DABS_MIGRATION_CHEATSHEET.md` → Pattern: Widget Parameters |
| Before/After examples | `README.md` → Common Issues Detected |
| Real-world example | `DATABRICKS_PATH_ALIGNMENT.md` |

---

## 🎓 Learning Path

### Beginner (Never used DABs before)

**Week 1: Understanding**
- [ ] Read `WORKFLOW_DIAGRAM.md`
- [ ] Read `SKILL_PACKAGE_SUMMARY.md`
- [ ] Review `DATABRICKS_PATH_ALIGNMENT.md` example

**Week 2: Practice**
- [ ] Run skill in `--mode analysis` on sample project
- [ ] Review generated documentation
- [ ] Study `DABS_MIGRATION_CHEATSHEET.md`

**Week 3: Execute**
- [ ] Run full migration on small project
- [ ] Use `MIGRATION_TEMPLATE.md` to track
- [ ] Deploy to dev environment

### Intermediate (Some DABs experience)

**Quick Start:**
- [ ] Run `/databricks-dabs-migration` on project
- [ ] Focus on issues you don't know how to fix
- [ ] Reference `DABS_MIGRATION_CHEATSHEET.md` for solutions

### Advanced (DABs expert)

**Use for:**
- [ ] Standardizing team migrations
- [ ] Catching edge cases
- [ ] Documenting patterns
- [ ] Training others

---

## 🔗 Cross-References

### Issue → Solution Mapping

| If you have... | Check these files... |
|----------------|---------------------|
| Type annotation errors | `README.md` #1, `CHEATSHEET.md` #1, `SUMMARY.md` Issue 1 |
| Hardcoded paths | `README.md` #2, `CHEATSHEET.md` #2, `SUMMARY.md` Issue 2 |
| Directory errors | `README.md` #3, `CHEATSHEET.md` #3, `SUMMARY.md` Issue 4 |
| MLflow errors | `README.md` #5, `CHEATSHEET.md` #4, `SUMMARY.md` Issue 5 |
| Library version issues | `README.md` #4, `CHEATSHEET.md` #5, `SUMMARY.md` Issue 3 |

### Concept → Documentation Mapping

| To learn about... | Read... |
|-------------------|---------|
| DABs parameter flow | `WORKFLOW_DIAGRAM.md`, `DATABRICKS_PATH_ALIGNMENT.md` |
| Directory structure | `CHEATSHEET.md`, `PATH_ALIGNMENT.md` |
| Migration process | `WORKFLOW_DIAGRAM.md`, `SUMMARY.md` |
| Code patterns | `CHEATSHEET.md`, `README.md` |
| Real-world example | `DATABRICKS_PATH_ALIGNMENT.md` |

---

## 📋 Checklists

### Pre-Migration Checklist
```
□ Read SKILL_PACKAGE_SUMMARY.md
□ Run /databricks-dabs-migration --mode analysis
□ Review generated documentation
□ Copy MIGRATION_TEMPLATE.md for tracking
□ Backup code to git
□ Plan migration timeline
```

### Migration Execution Checklist
```
□ Fix dependencies
□ Fix paths
□ Update DABs configuration
□ Fix code quality issues
□ Test locally (if possible)
□ Validate bundle
□ Deploy to dev
□ Test execution
□ Document issues
```

### Validation Checklist
```
□ databricks bundle validate passes
□ databricks bundle deploy succeeds
□ Jobs run without errors
□ Files in correct locations
□ MLflow experiments visible
□ Imports work
□ No hardcoded paths
□ All tests pass
```

---

## 🆘 Getting Help

### Order of Operations

1. **Check Cheat Sheet First**
   - `DABS_MIGRATION_CHEATSHEET.md`
   - Most common issues with quick solutions

2. **Search README**
   - `README.md` → Troubleshooting section
   - More detailed explanations

3. **Review Generated Docs**
   - Your project-specific issues
   - Customized recommendations

4. **Check Real Example**
   - `DATABRICKS_PATH_ALIGNMENT.md`
   - See how it was done in practice

5. **Ask Claude**
   - With specific error message
   - Reference the skill documentation

---

## 🎯 Success Indicators

### You know the skill is working when:

✅ Generated documentation is specific to your project
✅ Issues are clearly categorized and prioritized
✅ Solutions include code examples
✅ Migration guide has step-by-step instructions
✅ All edge cases are documented

### You know the migration succeeded when:

✅ `databricks bundle validate -t dev` passes
✅ `databricks bundle deploy -t dev` completes
✅ Jobs run without path errors
✅ Files created in expected locations
✅ MLflow experiments appear correctly
✅ Team can deploy independently

---

## 📝 Contributing

### To improve this skill:

1. **Found a new pattern?**
   - Add to `databricks-dabs-migration.md`
   - Update `CHEATSHEET.md`
   - Document in `README.md`

2. **Hit a new error?**
   - Add to troubleshooting sections
   - Create before/after example
   - Update relevant documentation

3. **Better way to explain?**
   - Update `WORKFLOW_DIAGRAM.md`
   - Enhance `SUMMARY.md`
   - Improve `README.md`

---

## 🗂️ File Organization

```
.claude/skills/
├── databricks-dabs-migration.md    # 📋 Main skill definition
├── README.md                       # 📖 User guide
├── DABS_MIGRATION_CHEATSHEET.md    # ⚡ Quick reference
├── MIGRATION_TEMPLATE.md           # 📊 Progress tracker
├── WORKFLOW_DIAGRAM.md             # 🎨 Visual guides
└── INDEX.md                        # 📚 This file

Project Root:
├── SKILL_PACKAGE_SUMMARY.md        # 📦 Complete overview
├── DATABRICKS_PATH_ALIGNMENT.md    # 🔍 Real-world example
└── Generated during migration:
    ├── MIGRATION_DISCOVERY.md
    ├── PATH_MIGRATION_PLAN.md
    ├── DEPENDENCY_MIGRATION_PLAN.md
    ├── CONFIG_REVIEW.md
    ├── CODE_QUALITY_FIXES.md
    ├── MIGRATION_GUIDE.md
    └── DATABRICKS_MIGRATION_COMPLETE.md
```

---

## 🚀 Next Steps

1. **If you haven't read anything yet:**
   → Start with `SKILL_PACKAGE_SUMMARY.md`

2. **If you're ready to migrate:**
   → Run `/databricks-dabs-migration`

3. **If you need a quick answer:**
   → Check `DABS_MIGRATION_CHEATSHEET.md`

4. **If you want to understand the process:**
   → Read `WORKFLOW_DIAGRAM.md`

5. **If you're validating work:**
   → Use `MIGRATION_TEMPLATE.md` to track

---

## 📞 Quick Links

| I need... | Go to... |
|-----------|----------|
| **To understand what this is** | `SKILL_PACKAGE_SUMMARY.md` |
| **To learn how to use it** | `README.md` |
| **To fix a specific error** | `DABS_MIGRATION_CHEATSHEET.md` |
| **To see the workflow** | `WORKFLOW_DIAGRAM.md` |
| **To track my progress** | `MIGRATION_TEMPLATE.md` |
| **To find everything** | `INDEX.md` (you are here!) |

---

**Version:** 1.0.0
**Last Updated:** 2026-01-25
**Status:** ✅ Complete and Ready to Use

---

## 🎉 Ready to Migrate!

You now have a complete skill package to help migrate any project to Databricks Asset Bundles.

**Start here:** `/databricks-dabs-migration`

Good luck! 🚀
