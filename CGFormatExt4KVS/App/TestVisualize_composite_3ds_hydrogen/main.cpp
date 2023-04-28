/*****************************************************************************/
/**
 *  @file   main.cpp
 *  @brief  Example program to visualize 3ds file.
 *  @author Insight, Inc.
 */
#include <kvs/Version>
#include <kvs/HydrogenVolumeData>
#include <kvs/CellByCellMetropolisSampling>
#include <kvs/ParticleBasedRenderer>
#include <kvs/StochasticRenderingCompositor>

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
    if(argc <= 5){
        std::cerr << "Set arguments, filepath, scale, translatex,y,z" << std::endl;
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
    
    char* filepath = argv[1];
    float scale = std::stof(argv[2]);
    float translate_x = std::stof(argv[3]);
    float translate_y = std::stof(argv[4]);
    float translate_z = std::stof(argv[5]);
    
    kvs::jaea::TexturedPolygonObject* texturedPolygon = KVSMain::CreateTexturedPolygonObject( argv[1] );
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
    
    kvs::jaea::StochasticTexturedPolygonRenderer* textured_polygon_renderer = new kvs::jaea::StochasticTexturedPolygonRenderer();
    textured_polygon_renderer->setEnabledTextureLinear(true);
    textured_polygon_renderer->setEnabledTextureMipmap(true);
    textured_polygon_renderer->setEnabledTextureMipmapLinear(true);
    textured_polygon_renderer->setEnabledTextureAnisotropy(true);
    
    kvs::Vector3ui resolution( 32, 32, 32 );
    kvs::StructuredVolumeObject* hydrogen_volume = new kvs::HydrogenVolumeData( resolution );
    
    const size_t repetitions = 4;
    const float step = 0.5f;
    const kvs::TransferFunction tfunc( 256 );
    kvs::PointObject* point_object = new kvs::CellByCellMetropolisSampling( hydrogen_volume, repetitions, step, tfunc );
    point_object->print( std::cout << std::endl );
    
    kvs::glsl::ParticleBasedRenderer* point_renderer = new kvs::glsl::ParticleBasedRenderer();
    
    screen.registerObject( point_object, point_renderer );
    screen.registerObject( texturedPolygon, textured_polygon_renderer );
    
    kvs::StochasticRenderingCompositor compositor( screen.scene() );
    compositor.setRepetitionLevel( 1 );
    compositor.enableLODControl();
    screen.setEvent( &compositor );
    
    screen.show();
    return app.run();
}
