#include <iostream>
#include <ejdb/ejdb.h>

#include "optionparser.h"
#include "ViolenceModel.h"

option::ArgStatus checkFileArg(const option::Option& option, bool msg);

// http://optionparser.sourceforge.net/
 enum  optionIndex { UNKNOWN, INDEX_FILE, TRAIN, CLEAR };
 const option::Descriptor usage[] =
 {
  {UNKNOWN, 0,"" , ""    ,option::Arg::None, "USAGE: example [options]\n\n"
                                             "Options:" },
  {INDEX_FILE, 0, "f", "index-file", &checkFileArg,     "--index-file <file_path>, -f <file_path>  Index the videos specified in file." },
  {CLEAR,      0, "c", "clear",      option::Arg::None, "--clear, -c  Clear the index store before respecting any other options." },
  {TRAIN,      0, "t", "train",      option::Arg::None, "--train, -t  Train the model with the existing index." },
  {0,0,0,0,0,0}
 };

 // Path to index file.
static boost::filesystem::path indexFilePath;

int main(int argc, char* argv[]) {

	// Set up command line arg parser.
	option::Stats  stats(usage, argc, argv);
	option::Option* options = new option::Option[stats.options_max];
	option::Option* buffer = new option::Option[stats.buffer_max];

	// Create the parser in GNU mode (true as first argument).
	option::Parser parser(true, usage, argc, argv, options, buffer);

	if ( parser.error() ) return 1;

	ViolenceModel vm;

	if ( options[CLEAR] ) {
		vm.clear();
	}

	// First clear the training store.
	vm.index("output.mp4");
	vm.index("output_copy.mp4");

	if ( options[TRAIN] ) {
		vm.train();
	}

    return 0;
}

option::ArgStatus checkFileArg(const option::Option& option, bool msg)
{
	// Only set the input file path if it hasn't been set yet and the argument isn't empty.
	if ( strcmp(option.arg, "") != 0 && indexFilePath.empty() ) {
		boost::filesystem::path filePath(option.arg);
		if ( boost::filesystem::exists(filePath) ) {
			indexFilePath = filePath;
			return option::ARG_OK;
		}
	}

	if ( msg ) {
		const char * pathCStr = option.arg ? option.arg : "<empty>";
		std::cout<<"Index file at path " << pathCStr << " does not exist. Aborting.\n";
		option::printUsage(std::cout, usage);
	}
	return option::ARG_ILLEGAL;
}
