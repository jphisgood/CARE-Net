"""Dataset-specific semantic label names used by GeoProto."""


def _prostate_label_names():
    return {
        0: "BG",
        1: "Bladder",
        2: "Bone",
        3: "Obturator_Internus",
        4: "Transition_Zone",
        5: "Central_Gland",
        6: "Rectum",
        7: "Seminal_Vesicle",
        8: "Neurovascular_Bundle",
    }


def get_label_names(dataset):
    label_names = {}

    if dataset == "CARDIAC_bssFP":
        label_names = {
            0: "BG",
            1: "LV-MYO",
            2: "LV-BP",
            3: "RV",
        }
    elif dataset == "CARDIAC_LGE":
        label_names = {
            0: "BG",
            1: "LV-MYO",
            2: "LV-BP",
            3: "RV",
        }
    elif dataset == "ABDOMEN_MR":
        label_names = {
            0: "BG",
            1: "LIVER",
            2: "RIGHT_KIDNEY",
            3: "LEFT_KIDNEY",
            4: "SPLEEN",
        }
    elif dataset == "ABDOMEN_CT":
        label_names = {
            0: "BG",
            1: "SPLEEN",
            2: "RIGHT_KIDNEY",
            3: "LEFT_KIDNEY",
            4: "GALLBLADDER",
            5: "ESOPHAGUS",
            6: "LIVER",
            7: "STOMACH",
            8: "AORTA",
            9: "INFERIOR_VENA_CAVA",
            10: "PORTAL_VEIN_AND_SPLENIC_VEIN",
            11: "PANCREAS",
            12: "RIGHT_ADRENAL_GLAND",
            13: "LEFT_ADRENAL_GLAND",
        }
    elif dataset in ("MI-PRO", "Prostate_NCI", "Prostate_UCLH"):
        label_names = _prostate_label_names()

    return label_names
