#ifndef _TEXTURED_POLYGON_IMPORTER_H_
#define _TEXTURED_POLYGON_IMPORTER_H_

#include <Core/Visualization/Importer/ImporterBase.h>
#include "TexturedPolygonObject.h"

namespace kvs{
namespace jaea{

class FBX;
class ThreeDS;

class TexturedPolygonImporter : public kvs::ImporterBase, public TexturedPolygonObject {
    kvsModule( kvs::jaea::TexturedPolygonImporter, Importer );
    kvsModuleBaseClass( kvs::ImporterBase );
    kvsModuleSuperClass( kvs::jaea::TexturedPolygonImporter );

public:
    TexturedPolygonImporter(){};
    virtual ~TexturedPolygonImporter(){};
    
    SuperClass* exec( const kvs::FileFormatBase* file_format ) override;

private:
    void import3DS(const kvs::jaea::ThreeDS* threeDS);
    void importFBX( const kvs::jaea::FBX* fbx);  
};

} // namespace jaea
} // kvs 

#endif // _THREE_DS_H_
