/*****************************************************************************/
/**
 *  @file   main.cpp
 *  @brief  Example program to visualize 3ds file.
 *  @author Insight, Inc.
 */
#include <kvs/Version>

#include "TexturedPolygonObject.h"
#include "TexturedPolygonImporter.h"
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
        
        kvs::FileFormatBase* io = new kvs::jaea::ThreeDS();
        kvs::jaea::TexturedPolygonImporter* importer = nullptr;
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
    
    kvs::jaea::TexturedPolygonObject* texturedPolygon = KVSMain::CreateTexturedPolygonObject( argv[1] );
    if( texturedPolygon == nullptr ){
        return 0;
    }
    
    kvs::jaea::StochasticTexturedPolygonRenderer* renderer = new kvs::jaea::StochasticTexturedPolygonRenderer();
    renderer->setEnabledTextureLinear(true);
    renderer->setEnabledTextureMipmap(true);
    renderer->setEnabledTextureMipmapLinear(true);
    renderer->setEnabledTextureAnisotropy(true);
    
    screen.registerObject( texturedPolygon, renderer );
    
    screen.show();
    return app.run();
}
