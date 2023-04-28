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
    
#ifdef USE_OCULUS
    kvs::oculus::jaea::Application app(argc, argv);
    kvs::oculus::jaea::Screen screen(&app);
    
    kvs::oculus::jaea::TouchController event( &screen );
    event.setRotationFactor( 5000.0f );
    event.setTranslationFactor( 5000.0f );
    event.setScalingFactor(30000.0f);
#else    
    kvs::glut::Application app( argc, argv );
    kvs::glut::Screen screen( &app );
    screen.setTitle("Example program for kvs::StochasticPolygonRenderer");
#endif
    
    for (int i=1; i+5<=argc;i+=5) {
        char* filepath = argv[i];
        float scale = std::stof(argv[i+1]);
        float translate_x = std::stof(argv[i+2]);
        float translate_y = std::stof(argv[i+3]);
        float translate_z = std::stof(argv[i+4]);
            
        kvs::jaea::StochasticTexturedPolygonRenderer* renderer = new kvs::jaea::StochasticTexturedPolygonRenderer();
        renderer->setEnabledTextureLinear(true);
        renderer->setEnabledTextureMipmap(true);
        renderer->setEnabledTextureMipmapLinear(true);
        renderer->setEnabledTextureAnisotropy(true);
        
        kvs::jaea::TexturedPolygonObject* texturedPolygon = KVSMain::CreateTexturedPolygonObject( argv[i] );
        if( texturedPolygon == nullptr ){
            return 0;
        }
        kvs::Matrix44<kvs::Real32> matrix(
                scale,     0,     0, translate_x,
                    0, scale,     0, translate_y,
                    0,     0, scale, translate_z,
                    0,     0,     0,           1
        );
        {
            //transform coords
            kvs::ValueArray<kvs::Real32> coords = texturedPolygon->coords();
            std::vector<kvs::Real32> coordsTransform;
            
            kvs::Vector4<kvs::Real32> v1;
            for(int i=0;i<coords.size();i+=3){
                float x = coords[i + 0];
                float y = coords[i + 1];
                float z = coords[i + 2];
                v1.set(x, y, z, 1.0f);
                
                kvs::Vector4<kvs::Real32> v2 = matrix * v1;
                coordsTransform.push_back( v2.x() );
                coordsTransform.push_back( v2.y() );
                coordsTransform.push_back( v2.z() );
            }
            texturedPolygon->setCoords( kvs::ValueArray<kvs::Real32>(coordsTransform) );
        }
        {
            kvs::ValueArray<kvs::Real32> coords = texturedPolygon->coords();
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
        screen.registerObject( texturedPolygon, renderer );
    }
    
    kvs::StochasticRenderingCompositor compositor( screen.scene() );
    compositor.setRepetitionLevel( 1 );
    compositor.enableLODControl();
    screen.setEvent( &compositor );
    
    screen.show();
    return app.run();
}
