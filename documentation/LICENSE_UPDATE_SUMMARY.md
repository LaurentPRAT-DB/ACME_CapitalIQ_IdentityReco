# License Implementation Summary

## Overview

The project has been updated with **Apache License 2.0** - a permissive open-source license commonly used for enterprise and Databricks projects.

## Files Created/Modified

### New License Files

1. **`LICENSE`** ✅
   - Full Apache License 2.0 text
   - Copyright 2026 Laurent Prat
   - Standard Apache License format

2. **`NOTICE`** ✅
   - Third-party dependency attributions
   - License summaries for dependencies
   - S&P Capital IQ data usage notice
   - Pre-trained model licensing notes

3. **`LICENSE_SUMMARY.md`** ✅
   - User-friendly license explanation
   - Third-party license table
   - Usage guidelines
   - Compliance checklist

4. **`LICENSE_UPDATE_SUMMARY.md`** ✅ (this file)
   - Summary of all licensing changes

### Updated Files

#### Documentation
- **`README.md`** ✅
  - Added License section
  - Updated Contact section with GitHub link

#### Configuration Files
- **`setup.py`** ✅
  - Added license header
  - Updated `license="Apache License 2.0"`
  - Updated author info: "Laurent Prat"
  - Updated email: "laurent.prat@databricks.com"
  - Updated URL to GitHub repository
  - Added license classifier

- **`pyproject.toml`** ✅
  - Added `license = {text = "Apache-2.0"}`
  - Added authors section
  - Added keywords
  - Added classifiers including license

#### Source Files (with license headers)
- **`src/config.py`** ✅
- **`src/utils/spark_utils.py`** ✅
- **`example_spark_connect.py`** ✅
- **`test_spark_connect.py`** ✅

## License Details

### Apache License 2.0 Key Features

**Permissions:**
- ✅ Commercial use
- ✅ Distribution
- ✅ Modification
- ✅ Patent use
- ✅ Private use

**Conditions:**
- ⚠️ License and copyright notice must be included
- ⚠️ State changes when modifying code
- ⚠️ Include NOTICE file if provided

**Limitations:**
- ❌ No trademark use
- ❌ No liability warranty
- ❌ No warranty provided

### Why Apache License 2.0?

1. **Enterprise-Friendly**: Widely accepted in enterprise environments
2. **Patent Protection**: Includes express patent grant
3. **Permissive**: Allows commercial and proprietary use
4. **Compatible**: Works with most other licenses
5. **Industry Standard**: Used by Apache Software Foundation, Databricks, and many others

## License Headers

All key Python files now include:

```python
# Copyright 2026 Laurent Prat
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

## Third-Party Dependencies

All major dependencies and their licenses are documented in the `NOTICE` file:

| License Type | Libraries |
|--------------|-----------|
| Apache 2.0 | Databricks SDK, PySpark, MLflow, Delta Lake, Transformers |
| MIT | sentence-transformers, python-dotenv |
| BSD-3-Clause | PyTorch, NumPy, scikit-learn, pandas |

## Important Notes

### S&P Capital IQ Data
- **NOT included** in this repository
- Users **must have** appropriate licenses from S&P Global
- Documented in NOTICE file

### Pre-trained Models
- BGE, DistilBERT, DBRX, Llama models require separate licenses
- Users must review model cards and licensing
- Documented in NOTICE file

### Databricks Platform
- Subject to Databricks Terms of Service
- Unity Catalog, Vector Search, Model Serving require appropriate licenses
- Documented in NOTICE file

## Compliance Checklist

For users and distributors:

- [x] LICENSE file created
- [x] NOTICE file created
- [x] README updated with license info
- [x] setup.py includes license metadata
- [x] pyproject.toml includes license metadata
- [x] License headers added to key source files
- [x] Third-party dependencies documented
- [x] Data licensing requirements noted
- [x] Model licensing requirements noted

## Usage Guidelines

### For End Users
1. Review LICENSE file
2. Comply with third-party licenses (see NOTICE)
3. Ensure S&P Capital IQ data licenses are in place
4. Review pre-trained model licenses

### For Distributors
1. Include LICENSE file
2. Include NOTICE file
3. Keep copyright notices
4. Document modifications
5. Comply with third-party licenses

### For Contributors
1. License headers required for new files
2. Update NOTICE for new dependencies
3. Document third-party code appropriately
4. Follow Apache License 2.0 terms

## File Structure

```
.
├── LICENSE                      # Apache License 2.0 full text
├── NOTICE                       # Third-party attributions
├── LICENSE_SUMMARY.md          # User-friendly explanation
├── LICENSE_UPDATE_SUMMARY.md   # This file
├── README.md                   # Updated with license info
├── setup.py                    # Updated with license metadata
├── pyproject.toml              # Updated with license metadata
└── src/
    ├── config.py               # License header added
    └── utils/
        └── spark_utils.py      # License header added
```

## References

- **Apache License 2.0**: https://www.apache.org/licenses/LICENSE-2.0
- **License FAQ**: https://www.apache.org/foundation/license-faq.html
- **SPDX Identifier**: Apache-2.0
- **OSI Approved**: Yes

## Contact

For licensing questions:
- **Author**: Laurent Prat
- **Email**: laurent.prat@databricks.com
- **GitHub**: [@LaurentPRAT-DB](https://github.com/LaurentPRAT-DB)

---

**Date**: 2026-01-23
**License**: Apache License 2.0
**Copyright**: Copyright 2026 Laurent Prat
**Status**: ✅ Complete
