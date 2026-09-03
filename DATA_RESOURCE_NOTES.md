# Data resource notes

## Verified representations

- The primary T-cell analysis used the public `GWCD4i.pseudobulk_merged.h5ad`, `GWCD4i.DE_stats.by_guide.h5mu` and `GWCD4i.DE_stats.by_donors.h5mu` objects, not a reconstructed cell-level matrix.
- The State analyses used official `eval_best` compact truth/prediction differential-expression tables. No local State refit or raw cell-level download is required for the reported analyses.
- The Tahoe full-transcriptome core contains 50 contexts, 93 shared drug-dose interventions, two plates, 9,500 pseudobulks, 25,695 primary genes and 13,390,662 retained cells.
- The Lea matrix is `GSE207049_31Mar21_all_runs_voom_resid.txt.gz`; its decompressed MD5 is `fc2e3e6d082ab6431155830c9a7767a7`, matching the Zenodo record.
- The ARCHS4 analysis range-extracted 1,026 GSE207049 samples from `human_gene_v2.5.h5` and aggregated them without outcome use to 340 paired LCLs and 10,110 matched GeneIDs. The full 47.9-GB HDF5 was not downloaded for checksum recomputation.
- Supplementary Table 4 contains 202,302 non-zero mapped weights across all 14 PROGENy pathways. Each pathway vector has unit L2 norm after mapping to the 25,695-gene primary axis.
- The final multimodal inventory excludes copy number because the public WGS copy-number matrix covered 39 contexts rather than the 49-context wiring panel. Methylation, metabolomics and unrelated generic-protein panels are not part of the final manuscript evidence.

## Redistribution boundaries

- CGC-authored code, orchestration and documentation are MIT licensed.
- State model artifacts and redistributed outputs remain subject to the Arc State Model Non-Commercial License and require prominent citation of the State paper. The underlying cytokine dataset is CC BY-NC 4.0.
- Tahoe-100M is linked under its public CC0-1.0 dataset terms. Arc Virtual Cell Atlas scripts are not copied because a repository license was not identified.
- Lea's Zenodo data are CC BY 4.0. Upstream author scripts without an explicit license are not copied.
- PROGENy weights are redistributed under Apache-2.0; the license and attribution are included separately and are not replaced by MIT.
- Reactome release-97 annotations and HGNC data are CC0.
- DepMap-generated mutation and Chronos files are linked with DepMap attribution. MCLP RPPA500 and Sanger proteomics are linked rather than redistributed because file-specific third-party redistribution terms were not exposed in the verified metadata.
- The current ARCHS4 v2.5 download page does not state a file-specific license, while the original paper and current source repository use different Creative Commons formulations. The HDF5 and extracted counts are therefore linked rather than redistributed pending owner clarification.

No public accession used in the manuscript is guessed. The unresolved public release URL and archival DOI are listed in `SUBMISSION_BLOCKERS.md`.
