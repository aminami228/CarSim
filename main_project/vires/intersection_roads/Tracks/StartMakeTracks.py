#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append("/home/cmu/Software/VTD.2.0/Runtime/Tools/RodDistro_2442_Rod64b_4.5.4/Tools")

import CopyAndCleanUp
import CreateTracks




try:
    CopyAndCleanUp.removeDB("/home/cmu/Software/VTD.2.0/Data/Projects/intersection_behavior/intersection_roads", "ComplexJunction")
    CreateTracks.createTracks(overlayName="ComplexJunction", buildRootDir="/tmp/Rod/DBGen", stdSetupFile="/home/cmu/Software/VTD.2.0/Runtime/Tools/RodDistro_2442_Rod64b_4.5.4/TileLib/SetupFilesPool/VTL/Full/TT_SETUP.DAT",
                          sigMode=0, projectDir="/home/cmu/Software/VTD.2.0/Data/Projects/intersection_behavior/intersection_roads", trkBaseDir="/home/cmu/Software/VTD.2.0/Data/Projects/intersection_behavior/intersection_roads/Tracks", stlBaseDir="/home/cmu/Software/VTD.2.0/Data/Projects/intersection_behavior/intersection_roads/Tracks",
                          trackParameters=["-fontScaleX", "1.00", "-fontScaleY", "1.00", "-vcolor", "-new_pwr", "-pwr_dim", "1.35", "-2d", "-exact_len", 
                           "-new_link", "-stripes", "-skip", "0"], optionalTrackArgs=["-vcolor"],
                          trackIDs=["2", "4", "17", "18", "20", "23", "26", "27", "28", "30", 
                           "1", "3", "5", "6", "7", "8", "10", "11", "15", "16", 
                           "19", "21", "22", "24", "25", "29", "34", "35", "36", "38", 
                           "39", "40", "41", "42", "43", "45", "46", "47"], trackRangeMap={}, trackSetupFilesMap={}, extensionsMap={},
                          trackAreaParameters=[], optionalTrackAreaArgs=["-vcolor"], areaTrackIDs=["34", "34", "38", "41", "45"], areaJuncIDs=["7", "3", "5", "1", "4"],
                          suffixStr="pass0", outputFormat="osgb",
                          customAlphaTexFilenames=[],
                          doConvertPartial=True, 
                          doIncludeTexInDB=True, doCreateIRDB=False, doCreateShadowDB=True, doCreateUncompressedTexDB=False, doCreateHDRDB=False,
                          doCopyFlts=False, doCopyIntermediateFiles=False, doPreDeleteBuildDir=True, doPostDeleteBuildDir=True, doSkipConversion=False, doDeleteTracks=False, doAbortOnFailure=False)

    sys.exit(0)

except Exception as e:
    print "[Error] Failed to create Database: " + str(e)

    errorLogfileName = "/tmp/Rod/LogBackend.txt"

    try:
        if( os.path.exists(errorLogfileName) ):
            if( os.access(errorLogfileName, os.W_OK) ):
                errorLogfile = open(errorLogfileName, "w")
                errorLogfile.write(str(e) + "\n")
            else: # No write permission (file most likely belongs a different user)
                os.remove(errorLogfileName)
                errorLogfile = open(errorLogfileName, "w")
                os.chmod(errorLogfileName, 0666)
                errorLogfile.write(str(e) + "\n")
        else:
            errorLogfile = open(errorLogfileName, "w")
            os.chmod(errorLogfileName, 0666)
            errorLogfile.write(str(e) + "\n")

    except Exception as fileException:
        print "[Error] Failed to update error logfile '%s': %s." % (errorLogfileName, str(fileException))


    sys.exit(1)
