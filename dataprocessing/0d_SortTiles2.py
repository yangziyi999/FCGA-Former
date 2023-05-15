import json
from glob import glob
import os
from argparse import ArgumentParser
import random
import numpy as np
from shutil import copyfile
import pandas as pd


# def extract_stage(metadata):
#     stage = metadata['cases'][0]['diagnoses'][0]['tumor_stage']
#     stage = stage.replace(" ", "_")
#     stage = stage.rstrip("a")
#     stage = stage.rstrip("b")
#     return stage
#
#
def extract_cancer(metadata):
    return metadata['cases'][0]['submitter_id']


#
#
def extract_sample_type(metadata):
    return metadata['cases'][0]['samples'][0]['sample_type']


#
#
# def sort_cancer_stage_separately(metadata, **kwargs):
#     sample_type = extract_sample_type(metadata) # 查找sample_type
#     cancer = extract_cancer(metadata) # 查找project_id
#     if "Normal" in sample_type:
#         stage = sample_type.replace(" ", "_")
#     else:
#         stage = extract_stage(metadata)
#
#     return os.path.join(cancer, stage) #返回id type
#
#
# def sort_cancer_stage(metadata, **kwargs):
#     sample_type = extract_sample_type(metadata)
#     cancer = extract_cancer(metadata)
#     stage = extract_stage(metadata)
#     if "Normal" in sample_type:
#         return sample_type.replace(" ", "_")
#     return cancer + "_" + stage  #返回id tumor_stage": "stage_i",
#
#
def sort_type(metadata, **kwargs):
    submitter = extract_cancer(metadata)
    d = pd.read_csv('/home/ps/SDW/sdwdata/TCGA_LUAD_merge.csv', usecols=['bcr_patient_barcode', 'PL_group'])
    sample_type = extract_sample_type(metadata)
    if "Normal" in sample_type:
        return sample_type.replace(" ", "_")
    else:
        index = d.values[np.argwhere(d.values[:, 0] == submitter), 1]
        # print('index', index)
        if (index.all() == 'Normal'):
            cancer = 'Normal'
        #elif (index.all() == 'Mid'):
            #cancer = 'Mid'
        elif (index.all() == 'High'):
            cancer = 'High'
        else:
            cancer = 'nan'
        return cancer


#
#
# def sort_cancer_type(metadata, **kwargs):
#     sample_type = extract_sample_type(metadata)
#     if "Normal" in sample_type:
#         return None
#     return extract_cancer(metadata)
#
#
# def sort_cancer_healthy_pairs(metadata, **kwargs):
#     sample_type = extract_sample_type(metadata)
#     cancer = extract_cancer(metadata)
#     if "Normal" in sample_type:
#         return os.path.join(cancer, sample_type.replace(" ", "_"))
#     return os.path.join(cancer, cancer)
#
#
# def sort_cancer_healthy(metadata, **kwargs):
#     sample_type = extract_sample_type(metadata)
#     if "Normal" in sample_type:
#         return sample_type.replace(" ", "_")
#     return "cancer"
#
#
# def sort_random(metadata, **kwargs):
#     AllOptions = ['TCGA-LUAD', 'TCGA-LUSC', 'Solid_Tissue_Normal']
#     return AllOptions[random.randint(0, 2)]
#
#
# def sort_mutational_burden(metadata, load_dic, **kwargs):
#     submitter_id = metadata["cases"][0]["submitter_id"]
#     try:
#         return load_dic[submitter_id]
#     except KeyError:
#         return None
#
#
# def sort_mutation_metastatic(metadata, load_dic, **kwargs):
#     sample_type = extract_sample_type(metadata)
#     if "Metastatic" in sample_type:
#         submitter_id = metadata["cases"][0]["submitter_id"]
#         try:
#             return load_dic[submitter_id]
#         except KeyError:
#             return None
#     return None
#
#
# def sort_setonly(metadata, load_dic, **kwargs):
#     return 'All'
#
#
# def sort_location(metadata, load_dic, **kwargs):
#     sample_type = extract_sample_type(metadata)
#     return sample_type.replace(" ", "_")
#
#
# def sort_melanoma_POD(metadata, load_dic, **kwargs):
#     Response = metadata['Response to Treatment (Best Response)']
#     if 'POD' in Response:
#         return 'POD'
#     else:
#         return 'Response'
#
#
# def sort_melanoma_Toxicity(metadata, load_dic, **kwargs):
#     return metadata['Toxicity Observed']
#
#
# def sort_text(metadata, load_dic, **kwargs):
#     return metadata
#
#
# def copy_svs_lymph_melanoma(metadata, load_dic, **kwargs):
#     sample_type = extract_sample_type(metadata)
#     if "Metastatic" in sample_type:
#         submitter_id = metadata["cases"][0]["diagnoses"][0]["tissue_or_organ_of_origin"]
#         if 'c77' in submitter_id:
#             try:
#                 return True
#             except KeyError:
#                 return False
#         else:
#             return False
#     return False
#
#
# def copy_svs_skin_primtumor(metadata, load_dic, **kwargs):
#     sample_type = extract_sample_type(metadata)
#     if "Primary" in sample_type:
#         submitter_id = metadata["cases"][0]["diagnoses"][0]["tissue_or_organ_of_origin"]
#         if 'c44' in submitter_id:
#             try:
#                 return True
#             except KeyError:
#                 return False
#         else:
#             return False
#     return False
#
#
# def sort_normal_txt(metadata, load_dic, **kwargs):
#     # sample_type = extract_sample_type(metadata)
#     # if "Normal" in sample_type:
#     #     return sample_type.replace(" ", "_")
#     # else:
#     #     submitter_id = metadata["cases"][0]["submitter_id"]
#     #     try:
#     #         return load_dic[submitter_id].replace(" ", "_")
#     #     except:
#     #         return None
#     #         #return False
#     submitter_id = metadata["cases"][0]["submitter_id"]
#     if submitter_id in load_dic.keys():
#         sample_type = extract_sample_type(metadata)
#         if "Normal" in sample_type:
#             return sample_type.replace(" ", "_")
#         else:
#             return load_dic[submitter_id].replace(" ", "_")
#     else:
#         return None
#
#
# def sort_melanoma_POD_Rec(metadata, load_dic, **kwargs):
#     Response = metadata['Response to Treatment (Best Response)']
#     if 'POD' in Response:
#         return 'POD'
#     elif 'PR' in Response:
#         return 'Response'
#     elif 'CR' in Response:
#         return 'Response'
#     else:
#         return None
#
#
# def sort_subfolders(metadata, load_dic, **kwargs):
#     return 'All'


sort_options = [

    sort_type,
]

if __name__ == '__main__':
    descr = """

    In this example, the images are expected to be in folders in this directory: '/ifs/data/abl/deepomics/pancreas/images_TCGA/512pxTiled_b'
    Each images should have its own sub-folder with the svs image name followed by '_files'
    Each images should have subfolders with names corresponding to the magnification associated with the jpeg tiles saved inside it
    The sorting will be done using tiles corresponding to a magnification of 20 (+/- 5 if the 20 folder does not exist)
    15%% will be put for validation, 15%% for testing and the leftover for training. However, if split is > 0, then the data will be split in train/test only in "# split" non-overlapping ways (each way will have 100/(#split) % of test images).
    linked images' names will start with 'train_', 'test_' or 'valid_' followed by the svs name and the tile ID
    """
    ## Define Arguments
    parser = ArgumentParser(description=descr)

    parser.add_argument("--SourceFolder", help="path to tiled images /home/ps/SDW/sdwdata/svs224/*", dest='SourceFolder',
                        default='/home/ps/SDW/sdwdata/svs512')
    parser.add_argument("--JsonFile", help="path to metadata json file", dest='JsonFile',
                        default='/home/ps/SDW/sdwdata/metadata.cart.2017-03-02T00_36_30.276824.json')
    parser.add_argument("--Magnification", help="magnification to use", type=float, dest='Magnification', default=20)
    parser.add_argument("--MagDiffAllowed", help="difference allowed on Magnification", type=float,
                        dest='MagDiffAllowed', default=0)
    parser.add_argument("--SortingOption", help="see option at the epilog", type=int, dest='SortingOption', default=3)
    parser.add_argument("--PercentTest", help="percentage of images for testing (between 0 and 100)", type=float,
                        dest='PercentTest', default=20)
    parser.add_argument("--PatientID",
                        help="Patient ID is supposed to be the first PatientID characters (integer expected) of the folder in which the pyramidal jpgs are. Slides from same patient will be in same train/test/valid set. This option is ignored if set to 0 or -1 ",
                        type=int, dest='PatientID', default=12)
    # parser.add_argument("--TMB", help="path to json file with mutational loads; or to BRAF mutations", dest='TMB')
    parser.add_argument("--nSplit", help="integer n: Split into train/test in n different ways", dest='nSplit',
                        default=0)
    parser.add_argument("--Balance",
                        help="balance datasets by: 0- tiles (default); 1-slides; 2-patients (must give PatientID)",
                        type=int, dest='Balance')
    # parser.add_argument("--outFilenameStats",
    #                     help="Check if the tile exists in an out_filename_Stats.txt file and copy it only if it True, or is the expLabel option had the highest probability",
    #                     dest='outFilenameStats')
    # parser.add_argument("--expLabel",
    #                     help="Index of the expected label within the outFilenameStats file (if only True/False is needed, leave this option empty). comma separated string expected",
    #                     dest='expLabel')
    # parser.add_argument("--threshold",
    #                     help="threshold above which the probability the class should be to be considered as true (if not specified, it would be considered as true if it has the max probability). comma separated string expected",
    #                     dest='threshold')

    ## Parse Arguments
    args = parser.parse_args()

    if args.JsonFile is None:
        print("No JsonFile found")
        args.JsonFile = ''

    if args.PatientID is None:
        print("PatientID ignored")
        args.PatientID = 0

    if args.Balance is None:
        args.Balance = 0
        print(args.Balance)

    if args.nSplit is None:
        args.nSplit = 0
    elif int(args.nSplit) > 0:
        args.PercentValid = 100 / int(args.nSplit)
        args.PercentTest = 0

    # if args.outFilenameStats is None:
    #     outFilenameStats_dict = {}

    SourceFolder = os.path.abspath(args.SourceFolder)
    print(SourceFolder)
    print(os.path.join(SourceFolder, "*_files"))
    imgFolders = glob(os.path.join(SourceFolder, "*_files"))
    # print(imgFolders)
    random.shuffle(imgFolders)  # randomize order of images

    JsonFile = args.JsonFile
    nameLength = -1
    if '.json' in JsonFile:
        with open(JsonFile) as fid:
            jdata = json.loads(fid.read())
        try:
            jdata = dict((jd['file_name'].replace('.svs', ''), jd) for jd in jdata)
        except:
            jdata = dict((jd['Patient ID'], jd) for jd in jdata)
        # print("jdata:")
        # print(jdata)
    Magnification = args.Magnification
    MagDiffAllowed = args.MagDiffAllowed

    SortingOption = args.SortingOption - 1  # transform to 0-based index
    try:
        sort_function = sort_options[0]
    except IndexError:
        raise ValueError("Unknown sort option")
    print("sort_function is %s" % sort_function)

    PercentTest = args.PercentTest / 100.
    if not 0 <= PercentTest <= 1:
        raise ValueError("PercentTest is not between 0 and 100")
    # Tumor mutational burden dictionary
    # TMBFile = args.TMB
    mut_load = {}

    ## Main Loop
    print("******************")
    Classes = {}
    NbrTilesCateg = {}
    PercentTilesCateg = {}
    NbrImagesCateg = {}
    PercentSlidesCateg = {}
    NbrPatientsCateg = {}
    PercentPatientsCateg = {}
    Patient_set = {}
    NbSlides = 0
    ttv_split = {}
    nbr_valid = {}
    # print(imgFolders)
    low_num = 0
    high_num = 0
    mid_num = 0
    for cFolderName in imgFolders:
        NbSlides += 1
        print("**************** starting %s" % cFolderName)
        imgRootName = os.path.basename(cFolderName)
        print(imgRootName)
        imgRootName = imgRootName.replace('_files', '')
        print(imgRootName)

        if args.SortingOption == 10:
            SubDir = os.path.basename(os.path.normpath(SourceFolder))

        else:
            try:
                image_meta = jdata[imgRootName]
            except KeyError:
                try:
                    image_meta = jdata[imgRootName[:nameLength]]
                except KeyError:
                    try:
                        image_meta = jdata[imgRootName[:args.PatientID]]
                    except KeyError:
                        print("file_name %s not found in metadata" % imgRootName[:args.PatientID])
                        continue
            print(image_meta)
            SubDir = sort_function(image_meta, load_dic=mut_load)
        print("SubDir is %s" % SubDir)
        # if (SubDir == 'Low'):
        #     low_num+=1
        #     if low_num==85:
        #         continue
        # elif (SubDir == 'Mid'):
        #     mid_num+=1
        #     if mid_num==85:
        #         continue
        # elif (SubDir == 'High'):
        #     high_num+=1
        #     if high_num==85:
        #         continue
        # print(low_num,high_num,mid_num)
        if int(args.nSplit) > 0:
            if SubDir is None:
                print("image not valid for this sorting option")
                continue
            # n-fold cross validation
            for nSet in range(int(args.nSplit)):
                SetDir = "set_" + str(nSet)
                if not os.path.exists(SetDir):
                    os.makedirs(SetDir, mode=0o777)
                # print("SubDir is still %s" % SubDir)
                if SubDir is None:
                    print("image not valid for this sorting option")
                    continue
                if not os.path.exists(os.path.join(SetDir, SubDir)):
                    os.makedirs(os.path.join(SetDir, SubDir))
        else:
            SetDir = ""
            if SubDir is None:
                print("image not valid for this sorting option")
                continue
            if not os.path.exists(SubDir): 
                        
                os.makedirs(SubDir,  mode=0o777)
                os.chmod(SubDir, 0o777)
        print("SubDir is still %s" % SubDir)
        try:
            Classes[SubDir].append(imgRootName)
        except KeyError:
            Classes[SubDir] = [imgRootName]

        # Check in the reference directories if there is a set of tiles at the desired magnification
        AvailMagsDir = [x for x in os.listdir(cFolderName)
                        if os.path.isdir(os.path.join(cFolderName, x))]
        # print(AvailMagsDir)
        AvailMags = tuple(float(x) for x in AvailMagsDir)
        # print(AvailMags)
        # check if the mag was known for that slide
        if max(AvailMags) < 0:
            print("Magnification was not known for that file.")
            continue
        mismatch, imin = min((abs(x - Magnification), i) for i, x in enumerate(AvailMags))
        # print(mismatch,imin)
        if mismatch <= MagDiffAllowed:
            AvailMagsDir = AvailMagsDir[imin]
        else:
            # No Tiles at the mag within the allowed range
            print("No Tiles found at the mag within the allowed range.")
            continue

        # Copy/symbolic link the images into the appropriate folder-type
        print("Symlinking tiles... for subdir %s" % SubDir)
        SourceImageDir = os.path.join(cFolderName, AvailMagsDir, "*")
        AllTiles = glob(SourceImageDir)
        # print('ALLtiles',AllTiles)
        if SubDir in NbrTilesCateg.keys():
            print("%s Already in dictionary" % SubDir)
            # print(SubDir)
        else:
            # print("Not yet in dictionary:")
            # print(SubDir)
            NbrTilesCateg[SubDir] = 0
            NbrTilesCateg[SubDir + "_train"] = 0
            NbrTilesCateg[SubDir + "_test"] = 0

            PercentTilesCateg[SubDir + "_train"] = 0
            PercentTilesCateg[SubDir + "_test"] = 0

            # print(PercentTilesCateg)
            NbrImagesCateg[SubDir] = 0
            NbrImagesCateg[SubDir + "_train"] = 0
            NbrImagesCateg[SubDir + "_test"] = 0

            PercentSlidesCateg[SubDir + "_train"] = 0
            PercentSlidesCateg[SubDir + "_test"] = 0

            NbrPatientsCateg[SubDir] = 0
            NbrPatientsCateg[SubDir + "_train"] = 0
            NbrPatientsCateg[SubDir + "_test"] = 0

            PercentPatientsCateg[SubDir + "_train"] = 0
            PercentPatientsCateg[SubDir + "_test"] = 0

        NbTiles = 0
        ttv = 'None'
        if len(AllTiles) == 0:
            continue
        for TilePath in AllTiles:
            TileName = os.path.basename(TilePath)
            print("TileName is %s" % TileName)
            NbTiles += 1
            if args.Balance == 1:
                # print("current percent in test, valid and ID (bal slide):" +  str(PercentSlidesCateg.get(SubDir + "_test"))+ "; " +str(PercentSlidesCateg.get(SubDir + "_valid")))
                # print(PercentTest, PercentValid)
                # print(PercentSlidesCateg.get(SubDir + "_test") < PercentTest)
                # print(PercentSlidesCateg.get(SubDir + "_valid") < PercentValid)

                # rename the images with the root name, and put them in train/test/valid
                if (PercentSlidesCateg.get(SubDir + "_test") <= PercentTest) and (PercentTest > 0):
                    ttv = "test"

                else:
                    ttv = "train"
            elif args.Balance == 2:
                # print("current percent in test, valid and ID (bal patient):" +  str(PercentPatientsCateg.get(SubDir + "_test"))+ "; " +str(PercentPatientsCateg.get(SubDir + "_valid")))
                # print(PercentPatientsCateg.get(SubDir + "_test"))
                # print(PercentPatientsCateg.get(SubDir + "_valid"))
                # print(PercentTest, PercentValid)
                # print(PercentPatientsCateg.get(SubDir + "_test") < PercentTest)
                # print(PercentPatientsCateg.get(SubDir + "_valid") < PercentValid)

                # rename the images with the root name, and put them in train/test/valid
                if (PercentPatientsCateg.get(SubDir + "_test") <= PercentTest) and (PercentTest > 0):
                    ttv = "test"
                else:
                    ttv = "train"
            else:
                # print("current percent in test, valid and ID (bal tile):" +  str(PercentTilesCateg.get(SubDir + "_test"))+ "; " +str(PercentTilesCateg.get(SubDir + "_valid")))
                # print(PercentTilesCateg.get(SubDir + "_test"))
                # print(PercentTilesCateg.get(SubDir + "_valid"))
                # print(PercentTest, PercentValid)
                # print(PercentTilesCateg.get(SubDir + "_test") < PercentTest)
                # print(PercentTilesCateg.get(SubDir + "_valid") < PercentValid)

                # rename the images with the root name, and put them in train/test/valid
                if (PercentTilesCateg.get(SubDir + "_test") <= PercentTest) and (PercentTest > 0):
                    ttv = "test"
                else:
                    ttv = "train"
            # If that patient had an another slide/scan already sorted, assign the same set to this set of images
            # print(ttv)
            # print(imgRootName[:args.PatientID])
            if int(args.nSplit) > 0:
                for nSet in range(int(args.nSplit)):
                    ttv_split[SubDir][nSet] = "train"

                if args.PatientID > 0:
                    Patient = imgRootName[:args.PatientID]
                    if Patient in Patient_set:
                        SetIndx = Patient_set[Patient]
                        tileNewPatient = False
                    else:
                        SetIndx = nbr_valid[SubDir].index(min(nbr_valid[SubDir]))
                        Patient_set[Patient] = SetIndx
                        tileNewPatient = True
                else:
                    try:
                        Patient = imgRootName
                    except:
                        Patient = imgRootName[:nameLength]
                    if Patient in Patient_set:
                        SetIndx = Patient_set[Patient]
                        tileNewPatient = False
                    else:
                        SetIndx = nbr_valid[SubDir].index(min(nbr_valid[SubDir]))
                        Patient_set[Patient] = SetIndx
                        tileNewPatient = True
                    # SetIndx = nbr_valid[SubDir].index(min(nbr_valid[SubDir]))
                    # tileNewPatient = True

                ttv_split[SubDir][SetIndx] = "test"
                if NbTiles == 1:
                    NewPatient = tileNewPatient

                if args.Balance == 1:
                    if NbTiles == 1:
                        nbr_valid[SubDir][SetIndx] = nbr_valid[SubDir][SetIndx] + 1
                elif args.Balance == 2:
                    if NewPatient:
                        if NbTiles == 1:
                            nbr_valid[SubDir][SetIndx] = nbr_valid[SubDir][SetIndx] + 1
                else:
                    nbr_valid[SubDir][SetIndx] = nbr_valid[SubDir][SetIndx] + 1

                # print(ttv_split[SubDir])
                # print(nbr_valid[SubDir])

                for nSet in range(int(args.nSplit)):
                    SetDir = "set_" + str(nSet)
                    NewImageDir = os.path.join(SetDir, SubDir, "_".join(
                        (ttv_split[SubDir][nSet], imgRootName, TileName)))  # all train initially
                    print(NewImageDir)
                    os.symlink(TilePath, NewImageDir)

            else:
                if args.PatientID > 0:
                    Patient = imgRootName[:args.PatientID]
                else:
                    Patient = imgRootName
                if True:

                    if Patient in Patient_set:
                        ttv = Patient_set[Patient]
                        if NbTiles == 1:
                            NewPatient = False
                    else:
                        Patient_set[Patient] = ttv
                        if NbTiles == 1:
                            NewPatient = True

                # print(ttv)

                NewImageDir = os.path.join(SubDir, "_".join((ttv, imgRootName, TileName)))  # all train initially
                if not os.path.lexists(NewImageDir):
                    os.symlink(TilePath, NewImageDir)
        #     # update stats
        #
        if ttv == "train":
            if NewPatient:
                NbrPatientsCateg[SubDir + "_train"] = NbrPatientsCateg[SubDir + "_train"] + 1
            NbrTilesCateg[SubDir + "_train"] = NbrTilesCateg.get(SubDir + "_train") + NbTiles
            NbrImagesCateg[SubDir + "_train"] = NbrImagesCateg[SubDir + "_train"] + 1
        elif ttv == "test":
            if NewPatient:
                NbrPatientsCateg[SubDir + "_test"] = NbrPatientsCateg[SubDir + "_test"] + 1
            NbrTilesCateg[SubDir + "_test"] = NbrTilesCateg.get(SubDir + "_test") + NbTiles
            NbrImagesCateg[SubDir + "_test"] = NbrImagesCateg[SubDir + "_test"] + 1

        else:
            continue
        NbrTilesCateg[SubDir] = NbrTilesCateg.get(SubDir) + NbTiles
        NbrImagesCateg[SubDir] = NbrImagesCateg.get(SubDir) + 1
        if NewPatient:
            NbrPatientsCateg[SubDir] = NbrPatientsCateg.get(SubDir) + 1

        print("New Patient: " + str(NewPatient))
        print("NbrPatientsCateg[SubDir]: " + str(NbrPatientsCateg[SubDir]))
        print("imgRootName: " + str(imgRootName))
        #
        PercentTilesCateg[SubDir + "_train"] = float(NbrTilesCateg.get(SubDir + "_train")) / float(
            NbrTilesCateg.get(SubDir))
        PercentTilesCateg[SubDir + "_test"] = float(NbrTilesCateg.get(SubDir + "_test")) / float(
            NbrTilesCateg.get(SubDir))
        PercentSlidesCateg[SubDir + "_train"] = float(NbrImagesCateg.get(SubDir + "_train")) / float(
            NbrImagesCateg.get(SubDir))
        PercentSlidesCateg[SubDir + "_test"] = float(NbrImagesCateg.get(SubDir + "_test")) / float(
            NbrImagesCateg.get(SubDir))
        PercentPatientsCateg[SubDir + "_train"] = float(NbrPatientsCateg.get(SubDir + "_train")) / float(
            NbrPatientsCateg.get(SubDir))
        PercentPatientsCateg[SubDir + "_test"] = float(NbrPatientsCateg.get(SubDir + "_test")) / float(
            NbrPatientsCateg.get(SubDir))

        print("Done. %d tiles linked to %s " % (NbTiles, SubDir))
        print("Train / Test  tiles sets for %s = %f %%  / %f %% " % (
            SubDir, PercentTilesCateg.get(SubDir + "_train"), PercentTilesCateg.get(SubDir + "_test"),
        ))
        print("Train / Test  slides sets for %s = %f %%  / %f %% " % (
            SubDir, PercentSlidesCateg.get(SubDir + "_train"), PercentSlidesCateg.get(SubDir + "_test"),
        ))
        if args.PatientID > 0:
            print("Train / Test  patients sets for %s = %f %%  / %f %% " % (
                SubDir, PercentPatientsCateg.get(SubDir + "_train"), PercentPatientsCateg.get(SubDir + "_test"),
            ))

    for k, v in sorted(Classes.items()):
        print('list of images in class %s :' % k)
        print(v)

    for k, v in sorted(NbrTilesCateg.items()):
        print(k, v)
    for k, v in sorted(PercentTilesCateg.items()):
        print(k, v)
    for k, v in sorted(NbrImagesCateg.items()):
        print(k, v)
    if args.PatientID > 0:
        for k, v in sorted(NbrPatientsCateg.items()):
            print(k, v)
