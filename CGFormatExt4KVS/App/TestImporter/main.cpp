#undef NDEBUG
#include <cassert>
#include <iostream>
#include <ThreeDS.h>
#include <FBX.h>
#include <TexturedPolygonImporter.h>

using namespace std;

class UnitTest {
protected:
  std::string _filename;
  kvs::FileFormatBase* _io;
  kvs::jaea::TexturedPolygonImporter* _importer;

public:
  UnitTest(const std::string filename){
    _filename = filename;
    const std::string ext = filename.substr(filename.length()-4, 4);

    // select importer
    if(ext == ".3ds"){
      _io = new kvs::jaea::ThreeDS();
    }else if(ext == ".fbx"){
      _io = new kvs::jaea::FBX();
    }else{
      std::cerr << "Unknown file format:" << filename << std::endl;
      throw "Unknown file format!";
    }

    _io->read(filename);
    if(_io->isSuccess()){
      std::cout << "loading " << _io->filename() << " succeed." << std::endl;
    }else{
      std::cerr << "loading " << _io->filename() << " failed." << std::endl;
    }

    _importer = new kvs::jaea::TexturedPolygonImporter();
    _importer->exec(_io);
  }

  virtual ~UnitTest(){
    delete _importer;
    delete _io;
  }

  std::string filename() const{
    return _filename;
  }

  virtual bool test() = 0;
};

class UnitTestCube3DS : public UnitTest{
public:
  using UnitTest::UnitTest;

  bool test() override {
    kvs::jaea::ThreeDS* threeDS_io = static_cast<kvs::jaea::ThreeDS*>(_io);

    auto coords = threeDS_io->coords();
    assert(coords.size() == 81); // 27 nodes * (x, y, z)

    auto normals = threeDS_io->normals();
    assert(normals.size() == 81); // 27 nodes * (nx, ny, nz)

    auto connections = threeDS_io->connections();
    assert(connections.size() == 36); // 12 triangles * 3 node-ids

    auto texture2DCoords = threeDS_io->texture2DCoords();
    assert(texture2DCoords.size() == 54); // 27 nodes * 2 coord

    auto textureIds = threeDS_io->textureIds();
    assert(textureIds.size() == 27); // for 27 nodes
    
    //auto textures = threeDS_io->textures();
    //assert(textures.size() == 1);
    auto textures = threeDS_io->colorArrays();
    assert(textures.size() == 1);

    assert(threeDS_io->imageWidths()[0] == 2048);
    assert(threeDS_io->imageHeights()[0] == 2048);
    //assert(t.internalFormat() == GL_RGBA8);

    return true;
  }
};

class UnitTestCubeFBX : public UnitTest{
public:
  using UnitTest::UnitTest;

  bool test() override {
    kvs::jaea::FBX* threeDS_io = static_cast<kvs::jaea::FBX*>(_io);

    auto coords = threeDS_io->coords();
    assert(coords.size() == 84); // 28 nodes * (x, y, z)

    auto normals = threeDS_io->normals();
    assert(normals.size() == 84); // 28 nodes * (nx, ny, nz)

    auto connections = threeDS_io->connections();
    assert(connections.size() == 36); // 12 triangles * 3 node-ids

    auto texture2DCoords = threeDS_io->texture2DCoords();
    assert(texture2DCoords.size() == 56); // 28 nodes * 2 coord

    auto textureIds = threeDS_io->textureIds();
    assert(textureIds.size() == 28); // for 28 nodes
    
    //auto textures = threeDS_io->textures();
    //assert(textures.size() == 1);

    //auto t = textures[0];
    //assert(t.width() == 2048);
    //assert(t.height() == 2048);
    //assert(t.internalFormat() == GL_RGBA8);

    auto textures = threeDS_io->colorArrays();
    assert(textures.size() == 2);

    assert(threeDS_io->imageWidths()[0] == 2048);
    assert(threeDS_io->imageHeights()[0] == 2048);

    return true;
  }
};

class UnitTestGeneral : public UnitTest{
public:
  using UnitTest::UnitTest;
  bool test() override {
    _importer->print(std::cout);
    return true;
  }
};

int main(int argc, char *argv[]){
  bool unitTest = false;
  if(argc < 2){
    unitTest = true;
  }

  if(unitTest){
    UnitTestCube3DS testCube3DS = UnitTestCube3DS(R"(..\..\samples\3ds\cube\Test1.3ds)");
    if(testCube3DS.test()){
      std::cout << "done: " << testCube3DS.filename() << std::endl;
    }
    UnitTestCubeFBX testCubeFBX = UnitTestCubeFBX(R"(..\..\samples\fbx\cube\Test1.fbx)");
    if(testCubeFBX.test()){
      std::cout << "done: " << testCubeFBX.filename() << std::endl;
    }
  }else{
    std::string path = argv[1];
    std::string  ext = path.substr(path.length() - 4);
    if(ext == ".3ds" || ext == ".fbx"){
      UnitTestGeneral testCubeFBX = UnitTestGeneral(path);
      if(testCubeFBX.test()){
        std::cout << "done: " << testCubeFBX.filename() << std::endl;
      }
    }else{
      std::cerr << "Unknown File type: " << ext << std::endl;
    }
  }

  return EXIT_SUCCESS;
}
