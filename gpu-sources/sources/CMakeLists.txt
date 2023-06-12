target_sources(gpu_lib PRIVATE
                utility/image.cu
                utility/image.cuhxx
                utility/png_utility.cu
                utility/choquet.cu
                utility/index.cu
                texture/featuresExtractionT.cu
                texture/similarityMeasuresT.cu
                color/similarityMeasuresC.cu)
