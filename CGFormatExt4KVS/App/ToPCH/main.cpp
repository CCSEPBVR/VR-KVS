#include <fstream>
#include <iostream>

#include <ThreeDS.h>
#include <FBX.h>
#include <TexturedPolygonImporter.h>

using namespace std;

class Converter {
protected:
  std::string _filename;
  kvs::FileFormatBase* _io;
  kvs::jaea::TexturedPolygonImporter* _importer;

public:
  Converter(const std::string filename){
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

  virtual ~Converter(){
    delete _importer;
    delete _io;
  }

  std::string filename() const{
    return _filename;
  }

  bool exportAsPCH(const std::string& path) {
    std::ofstream ofs(path);

    auto out = &ofs; 
    //auto out = &std::cout;

    auto coords = _importer->coords();
    int numNodes = (int)coords.size() / 3;

    *out << numNodes << std::endl;
    for(int i=0; i<numNodes; i++){
      auto node = _importer->coord(i);
      *out << node.x() << " " << node.y() << " " << node.z() << std::endl;
    }

    auto connections = _importer->connections();
    int numPatches = (int)_importer->numberOfConnections();

    if(connections.size() / 3 != numPatches){
      std::cerr << "Error: All patch should be triangle." << std::endl;
      return false;
    }
    
    *out << numPatches << std::endl;
    for(int i=0; i<numPatches; i++){
      // all patch is triangle
      // In pch format, normal direction is the right-hand thread when viewed the shape FROM OUTSIDE.
      int top = i*3;
      int node0 = connections[top];
      int node1 = connections[top + 2];
      int node2 = connections[top + 1];

      *out << node0 << " " << node1 << " " << node2 << std::endl;
    }

    return true;
  }

  bool exportAsSTL(const std::string& path) {
    std::ofstream ofs(path);

    auto out = &ofs; 
    //auto out = &std::cout;

    auto coords = _importer->coords();
    auto connections = _importer->connections();
    int numPatches = (int)_importer->numberOfConnections();

    if(connections.size() / 3 != numPatches){
      std::cerr << "Error: All patch should be triangle." << std::endl;
      return false;
    }
    
    *out << "solid shapes" << std::endl;
    for(int i=0; i<numPatches; i++){
      // all patch is triangle
      int top = i*3;
      int node0 = connections[top];
      int node1 = connections[top + 1];
      int node2 = connections[top + 2];

      auto coord0 = _importer->coord(node0);
      auto coord1 = _importer->coord(node1);
      auto coord2 = _importer->coord(node2);

      auto vec0 = coord1 - coord0;
      auto vec1 = coord2 - coord0;
      auto normal = vec0.cross(vec1);
      normal /= (float)normal.length();
      
      *out << "facet normal " << normal.x() << " " << normal.y() << " " << normal.z() << std::endl;
      *out << "outer loop" << std::endl;
      *out << "vertex " << coord0.x() << " " << coord0.y() << " " << coord0.z() << std::endl;
      *out << "vertex " << coord1.x() << " " << coord1.y() << " " << coord1.z() << std::endl;
      *out << "vertex " << coord2.x() << " " << coord2.y() << " " << coord2.z() << std::endl;
      *out << "endloop" << std::endl;
      *out << "endfacet" << std::endl;
    }
    *out << "endsolid shapes" << std::endl;

    return true;
  }
};

int main(int argc, char *argv[]){
  bool unitTest = false;
  if(argc < 3){
    return EXIT_SUCCESS;
  }

  std::string inPath = argv[1];
  std::string outPath = argv[2];

  std::string  inExt = inPath.substr(inPath.length() - 4);
  std::string  outExt = outPath.substr(outPath.length() - 4);
  if(inExt == ".3ds" || inExt == ".fbx"){
    Converter converter = Converter(inPath);
    if(outExt == ".stl"){
      if(converter.exportAsSTL(outPath)){
        std::cout << "done: " << converter.filename() << std::endl;
      }
    }else if(outExt == ".pch"){
      if(converter.exportAsPCH(outPath)){
        std::cout << "done: " << converter.filename() << std::endl;
      }
    }else{
      std::cerr << "Unknown File type: " << outExt << std::endl;
    }
  }else{
    std::cerr << "Unknown File type: " << inExt << std::endl;
  }

  return EXIT_SUCCESS;
}
