/*****************************************************************************/
/**
 *  @file   main.cpp
 *  @brief  Example program to visualize 3ds and fbx files.
 *  @author Insight, Inc.
 */
#include <kvs/Version>
#include <kvs/StochasticRenderingCompositor>

#include "TexturedPolygonObject.h"
#include "TexturedPolygonImporter.h"
#include "FBX.h"
#include "ThreeDS.h"
#include "StochasticTexturedPolygonRenderer.h"

//#define USE_OCULUS
#ifdef USE_OCULUS
#include <Application.h>
#include <Screen.h>
#include <TouchController.h>
#else
#include <kvs/glut/Application>
#include <kvs/glut/Screen>
#endif

namespace KVSMain {
    // ポリゴンオブジェクトを生成する関数
    kvs::jaea::TexturedPolygonObject* CreateTexturedPolygonObject( char *arg )
    {
        std::string filePath = arg;
        
        const std::string extension = filePath.substr(filePath.length()-4, 4);
        
        kvs::FileFormatBase* io = nullptr;
        kvs::jaea::TexturedPolygonImporter* importer = nullptr;
        
        // select importer
        if(extension == ".3ds"){
            io = new kvs::jaea::ThreeDS();
        }else if(extension == ".fbx"){
            io = new kvs::jaea::FBX();
        }else{
            std::cerr << "Unknown file format:" << filePath << std::endl;
            throw "Unknown file format!";
        }
        
        io->read(filePath);
        if(io->isSuccess()){
            std::cout << "loading " << io->filename() << " succeed." << std::endl;
            importer = new kvs::jaea::TexturedPolygonImporter();
            importer->exec(io);
        }else{
            std::cerr << "loading " << io->filename() << " failed." << std::endl;
        }
        
        delete io;
        
        return importer;
    }
}

int main(int argc, char *argv[])
{
    if(argc <= 1){
        std::cerr << "No arguments." << std::endl;
        return 0;
    }
    
    int sumNumPolygons = 0;
    int sumNumVertices = 0;
    int sumNumTextures = 0;
    int sumNumPixels = 0;
    float minTextureRate = 0;
    int nFiles = 0;
    for (int i=1; i<argc;i++) {
        char* filepath = argv[i];
        nFiles++;
        
        kvs::jaea::TexturedPolygonObject* texturedPolygon = KVSMain::CreateTexturedPolygonObject( argv[i] );
        if( texturedPolygon == nullptr ){
            return 0;
        }
        
        kvs::ValueArray<kvs::Real32> coords = texturedPolygon->coords();
        kvs::ValueArray<kvs::UInt32> connections = texturedPolygon->connections();
        std::map<kvs::UInt32, kvs::UInt32> map_id_to_image_width = texturedPolygon->mapIdToImageWidth();
        std::map<kvs::UInt32, kvs::UInt32> map_id_to_image_height = texturedPolygon->mapIdToImageHeight();
        int nVertices = coords.size() / 3;
        int nPolyogns = connections.size() / 3;
        
        int tmpSumNumPixels = 0;
        for(auto itr = map_id_to_image_width.begin(); itr != map_id_to_image_width.end(); itr++){
            int id = itr->first;
            int width = itr->second;
            int height = map_id_to_image_height[id];
            //std::cout << "# textureId=" << id << ", width=" << width << ", height=" << height << std::endl;
            sumNumTextures++;
            tmpSumNumPixels += width * height;
        }
        float textureRate = 1.0f * tmpSumNumPixels / nPolyogns;
        std::cout << "# nPolyogns=" << nPolyogns << ", nVertices=" << nVertices <<
            ", sumNumPixels=" << tmpSumNumPixels << ", textureRate=" << textureRate << 
            std::endl;
            
        sumNumPolygons += nPolyogns;
        sumNumVertices += nVertices;
        sumNumPixels += tmpSumNumPixels;
        
        if(i == 1){
            minTextureRate = textureRate;
        }else{
            if(minTextureRate > textureRate){
                minTextureRate = textureRate;
            }
        }
        
        int count = coords.size();
        float minx = 0;
        float miny = 0;
        float minz = 0;
        float maxx = 0;
        float maxy = 0;
        float maxz = 0;
        for(int iii=0;iii<count;iii+=3){
            float x = coords[iii + 0];
            float y = coords[iii + 1];
            float z = coords[iii + 2];
            if(iii==0){
                minx = x;
                miny = y;
                minz = z;
                maxx = x;
                maxy = y;
                maxz = z;
            }else{
                if(minx > x)minx = x;
                if(miny > y)miny = y;
                if(minz > z)minz = z;
                if(maxx < x)maxx = x;
                if(maxy < y)maxy = y;
                if(maxz < z)maxz = z;
            }
        }
        std::cout << "#boundingbox : " <<
                "min=(" << minx << ", " << miny << ", " << minz << "), " <<
                "max=(" << maxx << ", " << maxy << ", " << maxz << "), " <<
                "range=(" << (maxx - minx) << ", " << (maxy - miny) << ", " << (maxz - minz) << "), " <<
                std::endl;
    }
    
    std::cout << "#########################################" << std::endl;
    std::cout << "# nFiles=" << nFiles << ", sumNumPolygons=" << sumNumPolygons << ", sumNumVertices=" << sumNumVertices <<
        ", sumNumTextures=" << sumNumTextures << ", sumNumPixels=" << sumNumPixels << 
        ", minTextureRate=" << minTextureRate << 
        std::endl;
    std::cout << "#########################################" << std::endl;
    
    return 0;
}
