target_sources(gpu_lib PRIVATE
                utility/image.cpp
                utility/image.hxx
                utility/png_utility.cpp
                utility/choquet.cu
                texture/featuresExtractionT.cu
                texture/similarityMeasuresT.cu
                color/similarityMeasuresC.cu)