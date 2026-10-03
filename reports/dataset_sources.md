# Dataset sources and attribution

## PlantVillage

- Source repository: https://github.com/spMohanty/PlantVillage-Dataset
- The project already contains 54,305 PlantVillage leaf images across 38 class folders in its local data directory.
- The data are controlled-background images. Holdout scores on this collection should not be described as field accuracy.

## PlantDoc

- Authors' dataset repository: https://github.com/pratikkayal/PlantDoc-Dataset
- Paper: Davinder Singh et al., PlantDoc: A Dataset for Visual Plant Disease Detection, CoDS-COMAD 2020. https://arxiv.org/abs/1911.10317
- License stated by the authors: Creative Commons Attribution 4.0 International (CC BY 4.0). The local working copy preserves the upstream README and LICENSE.txt beneath final_data/data/plantdoc/PlantDoc-Dataset.
- The published dataset count is 2,598 images. The Windows-compatible local copy contains the available image files in its train and test folders; one filename with a question mark was normalized to an underscore during extraction.
- Training uses only categories that have a clear corresponding PlantVillage label. Ambiguous categories are excluded. The separate PlantDoc test folder is retained for a field-style evaluation report.
- Cite the PlantDoc authors and retain the CC BY 4.0 attribution when redistributing any dataset material.

## Tamil Nadu crop references

- Tamil Nadu Agricultural University, Zone-wise Contingency Crop Plan: https://agritech.tnau.ac.in/pdf/SWM25%20Contigencyplan.pdf
- Government of Tamil Nadu, Agriculture Policy Note 2021–22, including major crop groups and Tamil Nadu paddy seasons: https://agritech.tnau.ac.in/govt_schemes_services/pdf/2021/agri_e_pn_2021_21.pdf
- Crop names and season notes in the application are regional browsing guidance. They are not additional supervised labels for the crop classifier.

## Paddy Doctor not bundled

- Dataset authors and data description: https://github.com/paddydoc/paddy-doctor-dataset
- Dataset paper: https://arxiv.org/abs/2205.11108
- The authors describe field-collected paddy images from near Tirunelveli, Tamil Nadu. The authors' website directs users to IEEE DataPort for the full set; the 10,407-image Kaggle subset is subject to Kaggle competition rules and requires accepting those rules.
- This project does not download gated data automatically. The current crop-disease classifier has no paddy disease classes. Add an eligible paddy dataset under the data directory and retrain a rice-specific model before enabling those predictions.
